import json
import logging
from typing import Any, Dict, Optional

import grpc
from omegaconf import DictConfig, ListConfig, OmegaConf

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc


def _to_plain_dict(value) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


class ZKInferenceClient:
    def __init__(
        self,
        target: str,
        grpc_max_message_mb: int = 64,
        logger: Optional[logging.Logger] = None,
    ):
        self.target = target
        self.grpc_max_message_bytes = grpc_max_message_mb * 1024 * 1024
        self.logger = logger or logging.getLogger(__name__)

    def _channel(self):
        return grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", self.grpc_max_message_bytes),
                ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
            ],
        )

    def submit_inference_request(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        simplify_model: bool = False,
        input_shapes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the accepted request ID; poll status for preparation/proof completion."""
        input_shapes = _to_plain_dict(input_shapes)
        input_shapes_payload = json.dumps(input_shapes) if input_shapes is not None else ""

        self.logger.info(
            "Submitting request name=%s split_mode=%s ops_per_chunk=%s",
            name,
            split_mode,
            ops_per_chunk,
        )

        with self._channel() as channel:
            stub = pb_grpc.ZKJobServiceStub(channel)
            response = stub.SubmitInferenceRequest(
                pb.InferenceRequest(
                    name=name,
                    onnx_model_path=onnx_model_path,
                    input_data_path=input_data_path,
                    split_mode=split_mode,
                    ops_per_chunk=ops_per_chunk,
                    simplify_model=simplify_model,
                    input_shapes=input_shapes_payload,
                )
            )

        if not response.request_id:
            raise RuntimeError(
                f"Coordinator returned empty request_id for request '{name}'"
            )

        self.logger.info("Submitted request %s; coordinator scheduler: %s",
                         response.request_id, response.scheduler or "unknown (older coordinator)")
        return response.request_id

    def get_request_status(self, request_id: str):
        with self._channel() as channel:
            stub = pb_grpc.ZKJobServiceStub(channel)
            return stub.GetRequestStatus(pb.RequestStatusRequest(request_id=request_id))

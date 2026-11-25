import json
import logging
import sys
import threading
import time
from concurrent import futures
from typing import Dict, Optional

import grpc
import hydra
from omegaconf import DictConfig, OmegaConf

import zkinfer.proto.zkservice_pb2 as pb2
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.config.runtime import build_runtime_config, normalize_runtime_config
from zkinfer.runtime.coordinator import Coordinator, PreparationQueueFull
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import NodeResources
from zkinfer.runtime.rayon_config import configure_coordinator_rayon


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def validate_config(cfg: DictConfig) -> None:
    if cfg.storage.backend not in {"filesystem", "s3"}:
        raise ValueError("storage.backend must be either 'filesystem' or 's3'")
    if cfg.storage.backend == "s3" and not cfg.storage.s3_bucket:
        raise ValueError("storage.s3_bucket must be set when storage.backend=s3")


def parse_optional_json_dict(value: str) -> Optional[Dict]:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def to_pb_artifact(ref: Optional[ArtifactRef]) -> pb2.ArtifactRef:
    if ref is None or ref.is_empty:
        return pb2.ArtifactRef()
    return pb2.ArtifactRef(
        backend=ref.backend or "",
        uri=ref.uri or "",
        sha256=ref.sha256 or "",
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket or "",
    )


def from_pb_artifact(ref) -> Optional[ArtifactRef]:
    if ref is None or not getattr(ref, "backend", "") or not getattr(ref, "uri", ""):
        return None
    return ArtifactRef(
        backend=ref.backend,
        uri=ref.uri,
        sha256=ref.sha256,
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket,
    )


class ZKCoordinatorGrpcService(pb_grpc.ZKJobServiceServicer):
    """gRPC control-plane adapter around the core Coordinator."""

    def __init__(
        self,
        coordinator: Coordinator,
        logger: Optional[logging.Logger] = None,
    ):
        self.coordinator = coordinator
        self.logger = logger or logging.getLogger(__name__)

    def SubmitInferenceRequest(self, request, context):
        try:
            input_shapes = parse_optional_json_dict(request.input_shapes)
            request_id = self.coordinator.enqueue_request(
                name=request.name,
                onnx_model_path=request.onnx_model_path,
                input_data_path=request.input_data_path,
                split_mode=request.split_mode,
                ops_per_chunk=request.ops_per_chunk,
                simplify_model=request.simplify_model,
                input_shapes=input_shapes,
            )
            return pb2.InferenceRequestAck(request_id=request_id,
                                           scheduler=self.coordinator.scheduler_policy)
        except PreparationQueueFull as exc:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(exc))
        except json.JSONDecodeError as exc:
            self._set_error(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                exc,
                "Invalid input_shapes JSON",
            )
        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error submitting inference request",
            )
        return pb2.InferenceRequestAck(request_id="")

    def GetRequestStatus(self, request, context):
        try:
            status = self.coordinator.get_request_status(request.request_id)
            return pb2.RequestStatusResponse(
                request_id=status["request_id"],
                status=status["status"],
                message=status["message"],
                total_jobs=status["total_jobs"],
                queued_jobs=status["queued_jobs"],
                in_progress_jobs=status["in_progress_jobs"],
                completed_jobs=status["completed_jobs"],
                failed_jobs=status["failed_jobs"],
                scheduler=status["scheduler"],
            )
        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error fetching request status",
            )
            return pb2.RequestStatusResponse(
                request_id=request.request_id,
                status="unknown",
                message=str(exc),
            )

    def PollNode(self, request, context):
        try:
            node = NodeResources(
                node_id=request.node_id,
                memory_total_gb=request.memory_total_gb,
                memory_available_gb=request.memory_available_gb,
                cpu_total_cores=request.cpu_total_cores,
                cpu_available_cores=request.cpu_available_cores,
                active_job_ids=tuple(request.active_job_ids),
            )
            job = self.coordinator.poll_node(node)
            if job is None:
                return pb2.JobLease(job_available=False)

            return pb2.JobLease(
                job_available=True,
                lease_id=job.lease_id or "",
                request_id=job.inference_request_id,
                job_id=job.job_id,
                model=to_pb_artifact(job.model_ref),
                input=to_pb_artifact(job.input_ref),
                setup_cache=to_pb_artifact(job.setup_cache_ref),
                output_prefix=to_pb_artifact(job.output_prefix_ref),
                predicted_duration_s=job.predicted_duration_s or 0.0,
                predicted_peak_memory_gb=job.predicted_peak_memory_gb or 0.0,
                setup_key=job.artifact_key or "",
            )
        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error polling node",
            )
            return pb2.JobLease(job_available=False)

    def SubmitJobResult(self, request, context):
        try:
            perf_metrics = (
                json.loads(request.perf_metrics_json)
                if request.perf_metrics_json
                else None
            )
            ok = self.coordinator.submit_job_result(
                node_id=request.node_id,
                lease_id=request.lease_id,
                job_id=request.job_id,
                status=request.status,
                artifact_manifest_ref=from_pb_artifact(request.artifact_manifest),
                perf_metrics=perf_metrics,
                message=request.message,
            )
            return pb2.StatusAck(ok=ok)
        except json.JSONDecodeError as exc:
            self._set_error(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                exc,
                "Invalid perf_metrics_json JSON",
            )
            return pb2.StatusAck(ok=False)
        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error submitting job result",
            )
            return pb2.StatusAck(ok=False)

    def SendNodeHeartbeat(self, request, context):
        try:
            self.coordinator.record_node_heartbeat(
                node_id=request.node_id,
                active_job_ids=list(request.active_job_ids),
                status=request.status,
                message=request.message,
            )
            return pb2.HeartbeatAck(success=True)
        except Exception as exc:
            self._set_error(
                context,
                grpc.StatusCode.INTERNAL,
                exc,
                "Error recording node heartbeat",
            )
            return pb2.HeartbeatAck(success=False)

    def _set_error(self, context, code, exc: Exception, message: str) -> None:
        self.logger.error("%s: %s", message, exc, exc_info=True)
        context.set_code(code)
        context.set_details(f"{message}: {exc}")


def create_grpc_server(
    cfg: DictConfig,
    service: ZKCoordinatorGrpcService,
) -> grpc.Server:
    max_message_bytes = cfg.coordinator.grpc_max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.coordinator.grpc_thread_pool_size),
        options=[
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ],
    )
    pb_grpc.add_ZKJobServiceServicer_to_server(service, server)
    return server


def run_dead_node_monitor(
    coordinator: Coordinator,
    stop_event: threading.Event,
    logger: logging.Logger,
) -> None:
    interval_sec = max(1, int(coordinator.heartbeat_config.interval_sec))
    expected = time.monotonic() + interval_sec
    while not stop_event.wait(interval_sec):
        now = time.monotonic()
        lag = now - expected
        if lag > 2.0:
            logger.warning("Dead-node monitor delayed by %.1fs", lag)
        try:
            coordinator.check_for_dead_nodes()
        except Exception:
            logger.exception("Dead-node monitor failed")
        expected = time.monotonic() + interval_sec


@hydra.main(config_path="../config", config_name="config", version_base=None)
def serve(cfg: DictConfig) -> None:
    cfg = normalize_runtime_config(cfg)
    validate_config(cfg)
    runtime_config = build_runtime_config(cfg=cfg, process_role="coordinator")
    configure_coordinator_rayon(runtime_config.coordinator.rayon_threads)
    logger = setup_logger(
        "coordinator",
        log_file=f"{runtime_config.coordinator.logs_dir}/coordinator.log",
    )

    logger.info("Starting ZK Coordinator gRPC Service")
    logger.debug("Loaded config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    coordinator = Coordinator(runtime_config=runtime_config, logger=logger, resolved_config=cfg)
    service = ZKCoordinatorGrpcService(coordinator=coordinator, logger=logger)
    server = create_grpc_server(cfg, service)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=run_dead_node_monitor,
        args=(coordinator, stop_event, logger),
        name="dead-node-monitor",
        daemon=True,
    )

    bind_address = f"{cfg.coordinator.host}:{cfg.coordinator.port}"
    server.add_insecure_port(bind_address)
    server.start()
    monitor_thread.start()
    logger.info("Coordinator gRPC server running at %s", bind_address)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutdown requested; stopping coordinator")
    finally:
        stop_event.set()
        server.stop(5).wait()
        monitor_thread.join(timeout=5)
        coordinator.shutdown()
        logger.info("Coordinator stopped successfully")


if __name__ == "__main__":
    serve()

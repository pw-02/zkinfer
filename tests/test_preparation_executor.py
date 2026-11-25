"""Regression coverage for bounded preparation outside the synchronous RPC pool."""
import logging
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import grpc

from tests.test_serving_reliability import runtime_config
from zkinfer.proto import zkservice_pb2 as pb, zkservice_pb2_grpc as rpc
from zkinfer.runtime.coordinator import Coordinator
from zkinfer.runtime.coordinator_grpc import ZKCoordinatorGrpcService
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import ProofJob, RequestStatus


class PreparationExecutorTests(unittest.TestCase):
    def test_single_rpc_thread_remains_responsive_and_queue_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = runtime_config(Path(tmp))
            cfg = replace(cfg, coordinator=replace(
                cfg.coordinator, preparation_workers=1, preparation_max_pending=0,
                preparation_dispatch="streaming",
            ))
            owner = Coordinator(cfg, logging.getLogger("test-preparation-executor"))
            entered, release = threading.Event(), threading.Event()
            rpc_pool = ThreadPoolExecutor(max_workers=1)
            server = grpc.server(rpc_pool)
            rpc.add_ZKJobServiceServicer_to_server(ZKCoordinatorGrpcService(owner), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            channel = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = rpc.ZKJobServiceStub(channel)

            def build(request, on_plan, on_job_ready, **kwargs):
                on_plan(1)
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test preparation timed out")
                empty = ArtifactRef.empty()
                job = ProofJob(request.name, "part", request.request_id,
                               empty, empty, empty, empty)
                on_job_ready(job)
                return [job]

            try:
                with patch.object(owner.request_builder, "build_jobs", side_effect=build):
                    try:
                        request = pb.InferenceRequest(name="model", onnx_model_path="m.onnx",
                                                      input_data_path="i.json", split_mode="none",
                                                      ops_per_chunk=1)
                        ack = stub.SubmitInferenceRequest(request, timeout=3)
                        self.assertTrue(ack.request_id)
                        self.assertTrue(entered.wait(3))
                        status = stub.GetRequestStatus(
                            pb.RequestStatusRequest(request_id=ack.request_id), timeout=3)
                        self.assertEqual(status.status, "preparing")
                        self.assertFalse(stub.PollNode(pb.NodeStatus(node_id="node"), timeout=3).job_available)
                        self.assertTrue(stub.SendNodeHeartbeat(
                            pb.NodeHeartbeat(node_id="node"), timeout=3).success)
                        with self.assertRaises(grpc.RpcError) as rejected:
                            stub.SubmitInferenceRequest(request, timeout=3)
                        self.assertEqual(rejected.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                        self.assertEqual(len(owner.active_requests), 1)
                    finally:
                        release.set()
                        owner.preparation_pool.shutdown(wait=True)
                    self.assertTrue(owner.active_requests[ack.request_id].preparation_complete)
                    self.assertFalse(owner.scheduler.empty())
            finally:
                release.set()
                channel.close()
                server.stop(0).wait()
                owner.shutdown()
                rpc_pool.shutdown(wait=True)

    def test_batch_failure_is_reported_after_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = runtime_config(Path(tmp))
            owner = Coordinator(cfg, logging.getLogger("test-batch-preparation"))
            owner.preparation_dispatch = "batch"
            try:
                with patch.object(owner.request_builder, "build_jobs",
                                  side_effect=RuntimeError("calibration failed")), \
                        patch.object(owner, "_verify_request_if_enabled") as verify:
                    rid = owner.enqueue_request("model", "m.onnx", "i.json", "none", 1)
                    owner.shutdown()
                    request = owner.active_requests[rid]
                    self.assertEqual(request.request_status, RequestStatus.FAILED)
                    self.assertIn("calibration failed", request.error_message)
                    verify.assert_not_called()
            finally:
                owner.shutdown()


if __name__ == "__main__":
    unittest.main()

import csv
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from tests.test_serving_reliability import runtime_config
from zkinfer.profiling.run_metadata import save_process_metadata, copy_run_metadata
from zkinfer.runtime.coordinator import Coordinator
from zkinfer.runtime.coordinator_grpc import ZKCoordinatorGrpcService
from zkinfer.runtime.node.prover_task import StageAwareEZKLProofStages
from zkinfer.client.api import ZKInferenceClient
from zkinfer.proto import zkservice_pb2 as pb


class ObservabilityTests(unittest.TestCase):
    def fused_stages(self, root, version=1, fail=False):
        stages = StageAwareEZKLProofStages.__new__(StageAwareEZKLProofStages)
        stages.status_file = str(root / "status.txt")
        stages._transition_path = root / "stage_transitions.csv"
        stages._transition_started = time.perf_counter()
        stages.admission_metrics = {"stage_admission_wait_time(s)": 0.0}
        stages.logger = logging.getLogger("test")

        @contextmanager
        def hold(stage):
            yield {"wait_time_s": 0.1, "memory_gb": 0.0, "samples": 0, "source": "test"}

        stages.admission = SimpleNamespace(hold=hold)
        stages.srs_path = "srs"
        for attr in [
            "witness_path",
            "compiled_circuit_path",
            "vk_path",
            "pk_path",
            "proof_path",
        ]:
            setattr(stages, attr, attr)

        def native(**kwargs):
            if version:
                base = kwargs["stage_trace_elapsed_base_s"]
                with (root / "fused_stage_transitions.csv").open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["timestamp_unix_us", "elapsed_s", "stage", "event"])
                    for i, stage in enumerate(
                        ["KEY_GEN", "PROVING", "PERSISTING_KEYS", "PERSISTING_PROOF"]
                    ):
                        writer.writerow([int(time.time() * 1e6), base + i, stage, "enter"])
            else:
                self.assertNotIn("stage_trace_elapsed_base_s", kwargs)
            if fail:
                raise RuntimeError("native failed")
            return {"total_time_s": 4}

        stages._load_fused_ezkl = lambda: SimpleNamespace(
            stage_trace_version=version,
            setup_and_prove=native,
        )
        return stages

    def test_fused_events_close_wait_and_preserve_native_timestamps(self):
        with TemporaryDirectory() as tmp:
            stages = self.fused_stages(Path(tmp))
            stages.setup_and_compute_proof()
            with stages._transition_path.open() as file:
                events = list(csv.DictReader(file))
            self.assertEqual(
                [event["stage"] for event in events],
                [
                    "WAITING_FOR_FUSED_KEYGEN_PROVE",
                    "FUSED_KEYGEN_PROVE",
                    "KEY_GEN",
                    "PROVING",
                    "PERSISTING_KEYS",
                    "PERSISTING_PROOF",
                ],
            )
            self.assertEqual(
                float(events[-1]["elapsed_s"]) - float(events[2]["elapsed_s"]),
                3,
            )
            self.assertTrue(all(event["timestamp"].endswith("+00:00") for event in events))
            self.assertTrue(
                stages.admission_metrics["ezkl_fused_stage_trace_available"]
            )

    def test_old_wheel_does_not_leave_execution_labeled_waiting(self):
        with TemporaryDirectory() as tmp:
            stages = self.fused_stages(Path(tmp), version=0)
            (Path(tmp) / "fused_stage_transitions.csv").write_text("stale trace")
            stages.setup_and_compute_proof()
            self.assertFalse(
                stages.admission_metrics["ezkl_fused_stage_trace_available"]
            )
            with stages._transition_path.open() as file:
                self.assertEqual(
                    list(csv.DictReader(file))[-1]["stage"],
                    "FUSED_KEYGEN_PROVE",
                )

    def test_failed_native_call_preserves_partial_stage_events(self):
        with TemporaryDirectory() as tmp:
            stages = self.fused_stages(Path(tmp), fail=True)
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                stages.setup_and_compute_proof()
            self.assertIn("PROVING", stages._transition_path.read_text())

    def test_response_and_submit_log_use_coordinator_policy(self):
        coordinator = SimpleNamespace(
            scheduler_policy="resource_aware",
            enqueue_request=Mock(return_value="request"),
        )
        service = ZKCoordinatorGrpcService(coordinator)
        response = service.SubmitInferenceRequest(pb.InferenceRequest(), Mock())
        self.assertEqual(response.scheduler, "resource_aware")
        client = ZKInferenceClient("unused")
        with patch.object(client, "_channel"), patch(
            "zkinfer.client.api.pb_grpc.ZKJobServiceStub"
        ) as stub:
            stub.return_value.SubmitInferenceRequest.return_value = response
            with self.assertLogs(client.logger, level="INFO") as logs:
                client.submit_inference_request(
                    "model",
                    "model.onnx",
                    "input.json",
                    "none",
                    1,
                )
            self.assertIn(
                "coordinator scheduler: resource_aware",
                "\n".join(logs.output),
            )

    def test_metadata_resolves_values_and_copies_immutable_snapshot(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = runtime_config(root)
            resolved = OmegaConf.create(
                {
                    "budget": 8,
                    "reserve": "${budget}",
                    "coordinator": {"scheduling_policy": "resource_aware"},
                }
            )
            with patch(
                "zkinfer.profiling.run_metadata.code_revision",
                return_value={"commit": "abc", "dirty": False},
            ):
                snapshot = save_process_metadata(
                    root / "config",
                    "node",
                    cfg,
                    resolved,
                )
            resolved.budget = 16
            copy_run_metadata(snapshot, root / "run")
            metadata = json.loads(
                (root / "run" / "config" / snapshot.name).read_text()
            )
            self.assertEqual(metadata["resolved_config"]["reserve"], 8)
            self.assertEqual(metadata["code"]["commit"], "abc")
            self.assertEqual(
                metadata["runtime_config"]["node"]["node_id"],
                "test-node",
            )
            coordinator = Coordinator(
                cfg,
                logging.getLogger("test"),
                resolved_config=resolved,
            )
            try:
                self.assertEqual(
                    coordinator.get_request_status("unknown")["scheduler"],
                    coordinator.scheduler.policy,
                )
            finally:
                coordinator.shutdown()

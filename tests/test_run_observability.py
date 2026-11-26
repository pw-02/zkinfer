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
    def test_separate_stage_events(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = StageAwareEZKLProofStages.__new__(StageAwareEZKLProofStages)
            stages.status_file = str(root / "status.txt")
            stages._transition_path = root / "stage_transitions.csv"
            stages._transition_started = time.perf_counter()
            stages.admission_metrics = {"stage_admission_wait_time(s)": 0.0}
            @contextmanager
            def hold(stage):
                yield {"wait_time_s": 0.1, "memory_gb": 0.0, "samples": 0, "source": "test"}
            stages.admission = SimpleNamespace(hold=hold)
            stages._try_load_keys_from_cache = Mock(return_value=(False, 0))
            stages._upload_to_cache = Mock(return_value=0)
            stages.srs_path = "srs"
            stages.ezkl = SimpleNamespace(setup=Mock(), prove=Mock())
            for attr in ("witness_path", "compiled_circuit_path", "vk_path", "pk_path", "proof_path"):
                setattr(stages, attr, attr)
            stages.gen_keys()
            stages.compute_proof()
            with stages._transition_path.open() as file:
                events = list(csv.DictReader(file))
            self.assertEqual([event["stage"] for event in events],
                             ["WAITING_FOR_KEY_GEN", "KEY_GEN", "WAITING_FOR_PROVING", "PROVING"])
            stages.ezkl.setup.assert_called_once()
            stages.ezkl.prove.assert_called_once()

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

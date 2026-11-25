from __future__ import annotations

import logging
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

import zkinfer.proto.zkservice_pb2 as pb
from zkinfer.config.runtime import build_runtime_config
from zkinfer.runtime.coordinator import Coordinator
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest, JobStatus, NodeResources, ProofJob, RequestStatus
from zkinfer.runtime.node.prover_task import ProverTaskRunner
from zkinfer.runtime.node.supervisor import NodeSupervisor, to_pb_artifact


def runtime_config(root: Path):
    cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
    cfg.coordinator.runs_dir = str(root / "requests")
    cfg.coordinator.logs_dir = str(root / "logs")
    cfg.coordinator.state_db_path = str(root / "state.sqlite3")
    cfg.node.runs_dir = str(root / "node-runs")
    cfg.node.logs_dir = str(root / "node-logs")
    cfg.node.node_id = "test-node"
    cfg.node.runtime_state_dir = str(root / "runtime-state")
    cfg.storage.request_path = str(root / "requests-store")
    cfg.cache.path = str(root / "setup")
    cfg.cache.keys = True
    return build_runtime_config(cfg, process_role="coordinator")


def add_request(coordinator: Coordinator, setup_key="circuit"):
    request = InferenceRequest("model", "model.onnx", "input.json", "none", 1, "fifo")
    coordinator._create_request_dirs(request)
    empty = ArtifactRef.empty()
    job = ProofJob(
        "model",
        "partition",
        request.request_id,
        empty,
        empty,
        ArtifactRef("filesystem", coordinator.setup_store.root_dir),
        empty,
        predicted_peak_memory_gb=1,
        artifact_key=setup_key,
        profiling_file_path=str(
            Path(coordinator.setup_store.root_dir) / "resource_profile.json"
        ),
    )
    request.proof_jobs = [job]
    coordinator._enqueue_request(request)
    return request, job


class ResultAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = runtime_config(Path(self.tmp.name))
        self.coordinator = Coordinator(
            self.cfg,
            logging.getLogger("test-coordinator"),
        )
        self.addCleanup(self.coordinator.shutdown)
        self.request, self.job = add_request(self.coordinator)
        self.node = NodeResources(
            "node",
            memory_total_gb=64,
            memory_available_gb=60,
        )
        self.coordinator.poll_node(self.node)

    def test_missing_stale_and_nonterminal_results_are_rejected(self):
        lease_id = self.job.lease_id
        for status, lease in (
            ("COMPLETED", ""),
            ("COMPLETED", "old"),
            ("QUEUED", lease_id),
        ):
            self.assertFalse(
                self.coordinator.submit_job_result(
                    "node",
                    self.job.job_id,
                    status,
                    lease,
                )
            )
            self.assertEqual(self.job.job_status, JobStatus.IN_PROGRESS)

    def test_accepted_completion_is_idempotent(self):
        lease = self.job.lease_id
        manifest = ArtifactRef("filesystem", "/attempt-a/manifest.json")
        with patch.object(self.coordinator, "_finalize_job"), \
                patch.object(self.coordinator, "_schedule_request_finalization"):
            self.assertTrue(
                self.coordinator.submit_job_result(
                    "node",
                    self.job.job_id,
                    "COMPLETED",
                    lease,
                    manifest,
                )
            )
            self.assertTrue(
                self.coordinator.submit_job_result(
                    "node",
                    self.job.job_id,
                    "COMPLETED",
                    lease,
                    manifest,
                )
            )

    def test_old_lease_cannot_complete_requeued_job(self):
        old_lease = self.job.lease_id
        with self.coordinator.lock:
            self.coordinator._requeue_active_job_locked(self.job.job_id)
        self.coordinator.poll_node(self.node)
        self.assertNotEqual(old_lease, self.job.lease_id)
        self.assertFalse(
            self.coordinator.submit_job_result(
                "node",
                self.job.job_id,
                "COMPLETED",
                old_lease,
            )
        )

    def test_terminal_completion_survives_restart(self):
        lease = self.job.lease_id
        manifest = ArtifactRef("filesystem", "/attempt-a/manifest.json")
        with patch.object(self.coordinator, "_finalize_job"), \
                patch.object(self.coordinator, "_schedule_request_finalization"):
            self.coordinator.submit_job_result(
                "node",
                self.job.job_id,
                "COMPLETED",
                lease,
                manifest,
            )
        self.request.request_status = RequestStatus.COMPLETED
        self.request.completed_time = datetime.now(timezone.utc)
        self.coordinator.state_store.save_request(self.request)
        restarted = Coordinator(self.cfg, logging.getLogger("test-restarted"))
        self.addCleanup(restarted.shutdown)
        self.assertTrue(
            restarted.submit_job_result(
                "node",
                self.job.job_id,
                "COMPLETED",
                lease,
                manifest,
            )
        )


class StreamingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = runtime_config(Path(self.tmp.name))
        self.coordinator = Coordinator(self.cfg, logging.getLogger("test-streaming"))
        self.addCleanup(self.coordinator.shutdown)
        self.coordinator.preparation_dispatch = "streaming"
        self.node = NodeResources("node", memory_total_gb=64, memory_available_gb=60)

    @staticmethod
    def job(request, name):
        empty = ArtifactRef.empty()
        return ProofJob(request.name, name, request.request_id, empty, empty, empty, empty)

    def submit(self):
        return self.coordinator.submit_request("model", "model.onnx", "input.json", "none", 1)

    def test_first_job_can_finish_before_last_job_is_published(self):
        owner = self.coordinator

        def build(request, on_plan, on_job_ready, **_kwargs):
            on_plan(2)
            first = self.job(request, "first")
            on_job_ready(first)
            self.assertEqual(owner.get_request_status(request.request_id)["total_jobs"], 2)
            self.assertIs(owner.poll_node(self.node), first)
            self.assertTrue(owner.submit_job_result("node", first.job_id, "COMPLETED", first.lease_id))
            self.assertFalse(request.all_jobs_finished())
            self.assertNotIn(request.request_id, owner.finalizing_requests)
            self.assertIsNone(request.completed_time)
            second = self.job(request, "second")
            on_job_ready(second)
            return [first, second]

        with patch.object(owner, "_build_prepared_jobs", side_effect=build), \
                patch.object(owner, "_finalize_job"), \
                patch.object(owner, "_verify_request_if_enabled", return_value={"valid": True}) as verify:
            rid = self.submit()
            request = owner.active_requests[rid]
            self.assertTrue(request.preparation_complete)
            self.assertFalse(request.all_jobs_finished())
            verify.assert_not_called()
            second = owner.poll_node(self.node)
            self.assertEqual(second.model_name, "second")
            owner.submit_job_result("node", second.job_id, "COMPLETED", second.lease_id)
            owner.finalization_pool.shutdown(wait=True)
            self.assertEqual(request.request_status, RequestStatus.COMPLETED)
            verify.assert_called_once()

    def test_preparation_failure_never_verifies_partial_graph(self):
        owner = self.coordinator

        def build(request, on_plan, on_job_ready, **_kwargs):
            on_plan(2)
            first = self.job(request, "first")
            on_job_ready(first)
            self.assertIs(owner.poll_node(self.node), first)
            owner.submit_job_result("node", first.job_id, "COMPLETED", first.lease_id)
            raise RuntimeError("calibration failed")

        with patch.object(owner, "_build_prepared_jobs", side_effect=build), \
                patch.object(owner, "_finalize_job"), \
                patch.object(owner, "_verify_request_if_enabled") as verify:
            rid = self.submit()
            owner.finalization_pool.shutdown(wait=True)
            request = owner.active_requests[rid]
            self.assertEqual(request.request_status, RequestStatus.FAILED)
            self.assertIn("calibration failed", request.error_message)
            verify.assert_not_called()

    def test_recovery_rejects_interrupted_preparation(self):
        owner = self.coordinator
        request = InferenceRequest("model", "model.onnx", "input.json", "none", 1, "fifo")
        owner._create_request_dirs(request)
        request.preparation_complete = False
        request.expected_job_count = 2
        request.request_status = RequestStatus.PREPARING
        job = self.job(request, "first")
        job.job_status = JobStatus.QUEUED
        owner.state_store.save_request(request)
        owner.state_store.save_job(job)
        restarted = Coordinator(self.cfg, logging.getLogger("test-streaming-recovery"))
        self.addCleanup(restarted.shutdown)
        restarted.finalization_pool.shutdown(wait=True)
        recovered = restarted.active_requests[request.request_id]
        self.assertEqual(recovered.request_status, RequestStatus.FAILED)
        self.assertIn("restarted", recovered.error_message)
        self.assertTrue(restarted.scheduler.empty())


class NodeAttemptTests(unittest.TestCase):
    def test_retry_replaces_local_job_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = NodeSupervisor(runtime_config(root))
            lease = pb.JobLease(
                request_id="request",
                job_id="job",
                lease_id="lease-a",
            )
            _, tmp_dir, output_dir = node._job_paths(lease)
            tmp_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (output_dir / "old.csv").write_text("old")

            model = root / "model.onnx"
            input_file = root / "input.json"
            model.write_bytes(b"model")
            input_file.write_text('{"input_data": [[1]]}')
            lease.model.CopyFrom(to_pb_artifact(ArtifactRef("filesystem", str(model))))
            lease.input.CopyFrom(to_pb_artifact(ArtifactRef("filesystem", str(input_file))))
            lease.output_prefix.CopyFrom(
                to_pb_artifact(ArtifactRef("filesystem", str(root / "published")))
            )
            task = node.prepare_local_task(lease)
            self.assertFalse((Path(task.output_dir) / "old.csv").exists())

    def test_failed_job_removes_local_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = runtime_config(root)
            cfg = cfg.__class__(
                coordinator=cfg.coordinator,
                node=cfg.node,
                request_store=cfg.request_store,
                setup_store=cfg.setup_store.__class__(
                    enabled=False,
                    backend=cfg.setup_store.backend,
                    root_dir=cfg.setup_store.root_dir,
                    s3_bucket=cfg.setup_store.s3_bucket,
                    s3_prefix=cfg.setup_store.s3_prefix,
                ),
                jobs=cfg.jobs,
                heartbeat=cfg.heartbeat,
                state_store=cfg.state_store,
                verification=cfg.verification,
            )
            node = NodeSupervisor(cfg)
            lease = pb.JobLease(
                request_id="request",
                job_id="job",
                lease_id="lease",
            )
            request_dir, tmp_dir, output_dir = node._job_paths(lease)

            def prepare(*_args):
                tmp_dir.mkdir(parents=True)
                output_dir.mkdir(parents=True)
                (request_dir / "config").mkdir(parents=True)
                return SimpleNamespace(
                    tmp_dir=str(tmp_dir),
                    output_dir=str(output_dir),
                    setup_key="key",
                )

            with patch.object(node, "prepare_local_task", side_effect=prepare), \
                    patch.object(
                        node,
                        "_run_prover_subprocess",
                        side_effect=RuntimeError("failed"),
                    ), \
                    patch.object(node, "submit_job_result"), \
                    patch.object(node, "_cleanup_published_attempt") as cleanup:
                node.run_lease(lease)
                cleanup.assert_called_once()
            self.assertFalse(output_dir.exists())
            self.assertFalse(tmp_dir.exists())

    def test_directory_ids_cannot_escape_job_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = NodeSupervisor(runtime_config(Path(tmp)))
            for value in ("..", "../outside", "a/b", "a\\b", ""):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    node._job_paths(
                        pb.JobLease(
                            request_id="request",
                            job_id=value,
                            lease_id="lease",
                        )
                    )

    def test_published_files_are_lease_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "proof"
            source.write_bytes(b"proof")
            task = SimpleNamespace(
                lease_id="lease",
                output_backend="filesystem",
                output_prefix=str(root / "published"),
                output_s3_bucket="",
            )
            uri, _ = ProverTaskRunner()._publish_file(
                str(source),
                task,
                "proof.pf",
            )
            self.assertEqual(Path(uri).name, "lease--proof.pf")


if __name__ == "__main__":
    unittest.main()

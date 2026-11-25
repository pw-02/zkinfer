from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zkinfer.proto.zkservice_pb2 as pb
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import NodeResources, ProofJob
from zkinfer.runtime.node.stage_admission import (
    StageAdmissionController,
    StageResourceHistory,
)
from zkinfer.runtime.resource_profiles import (
    get_profile_estimates,
    update_resource_profile,
)
from zkinfer.runtime.scheduler import JobScheduler
from zkinfer.runtime.state_store import CoordinatorStateStore


def _job(request_id: str, name: str) -> ProofJob:
    empty = ArtifactRef.empty()
    return ProofJob(
        inference_request_name=request_id,
        model_name=name,
        inference_request_id=request_id,
        model_ref=empty,
        input_ref=empty,
        setup_cache_ref=empty,
        output_prefix_ref=empty,
        predicted_peak_memory_gb=1.0,
    )


def _wait_on_stage_admission(root: str, output: mp.Queue) -> None:
    controller = StageAdmissionController(
        root_dir=root,
        setup_key="setup-a",
        enabled=True,
        fallback_memory_gb=0.01,
        memory_safety_reserve_gb=0,
        poll_interval_s=0.02,
    )
    started = time.perf_counter()
    with patch("zkinfer.runtime.node.stage_admission.psutil.virtual_memory", return_value=_memory()), \
            patch("zkinfer.runtime.node.stage_admission.psutil.pid_exists", side_effect=_pid_exists), \
            patch("zkinfer.runtime.node.stage_admission.process_tree_memory_gb", return_value=0.0):
        with controller.hold("PROVING") as info:
            output.put((info["wait_time_s"], time.perf_counter() - started))


def _memory(total_gb=64.0, available_gb=60.0):
    return SimpleNamespace(total=total_gb * 1024**3, available=available_gb * 1024**3)


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class ProtocolTests(unittest.TestCase):
    def test_client_does_not_select_scheduler_per_request(self):
        fields = pb.InferenceRequest.DESCRIPTOR.fields_by_name
        self.assertNotIn("scheduler", fields)

    def test_node_status_has_no_job_slot_limit(self):
        fields = pb.NodeStatus.DESCRIPTOR.fields_by_name
        self.assertNotIn("max_concurrent_jobs", fields)


class SchedulerTests(unittest.TestCase):
    def test_zero_available_memory_rejects_known_demand(self):
        for policy in ("fifo", "lpt", "cache_aware", "resource_aware"):
            with self.subTest(policy=policy):
                scheduler = JobScheduler(policy)
                scheduler.add_jobs([_job("a", "a1")])
                self.assertIsNone(scheduler.next_job(NodeResources(
                    node_id="node", memory_total_gb=64, memory_available_gb=0,
                )))

    def test_new_requests_do_not_receive_a_dispatch_count_catchup_burst(self):
        scheduler = JobScheduler("fifo")
        scheduler.add_jobs([_job("old", f"o{i}") for i in range(12)])
        node = NodeResources(node_id="node", memory_available_gb=64)
        for _ in range(5):
            scheduler.next_job(node)
        scheduler.add_jobs([_job("new", f"n{i}") for i in range(5)])
        self.assertEqual(
            [scheduler.next_job(node).inference_request_id for _ in range(4)],
            ["old", "new", "old", "new"],
        )

    def test_infeasible_request_does_not_block_round_robin(self):
        scheduler = JobScheduler("fifo")
        big = _job("big", "b1")
        big.predicted_peak_memory_gb = 100
        scheduler.add_jobs([big, _job("small", "s1")])
        self.assertEqual(scheduler.next_job(NodeResources(
            node_id="node", memory_available_gb=10,
        )).inference_request_id, "small")

    def test_requests_rotate_before_one_request_monopolizes_queue(self):
        scheduler = JobScheduler("fifo")
        scheduler.add_jobs(
            [
                _job("request-a", "a1"),
                _job("request-a", "a2"),
                _job("request-a", "a3"),
                _job("request-b", "b1"),
            ]
        )
        node = NodeResources(node_id="node", memory_available_gb=64)
        first = scheduler.next_job(node)
        second = scheduler.next_job(node)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.inference_request_id, second.inference_request_id)

    def test_active_jobs_do_not_impose_a_slot_limit(self):
        scheduler = JobScheduler("fifo")
        scheduler.add_jobs([_job("request-a", "a1")])
        node = NodeResources(
            node_id="node",
            memory_available_gb=64,
            active_job_ids=("already-running",),
        )
        self.assertIsNotNone(scheduler.next_job(node))

    def test_stage_profile_allows_job_to_start_before_peak_memory_is_free(self):
        job = _job("request-a", "a1")
        job.predicted_peak_memory_gb = 40.0
        job.profiling_data = update_resource_profile(
            {},
            {
                "job_runtime(s)": 60.0,
                "max_process_memory(GB)": 40.0,
                "stage_generating_witness_max_memory(GB)": 6.0,
                "stage_generating_witness_avg_cpu_machine(%)": 20.0,
                "stage_proving_max_memory(GB)": 40.0,
                "stage_proving_avg_cpu_machine(%)": 80.0,
            },
            setup_key="setup-a",
        )
        scheduler = JobScheduler("resource_aware")
        scheduler.add_jobs([job])
        node = NodeResources(
            node_id="node",
            memory_total_gb=64.0,
            memory_available_gb=10.0,
        )
        self.assertIsNotNone(scheduler.next_job(node))

    def test_job_is_rejected_when_profile_peak_exceeds_machine_memory(self):
        job = _job("request-a", "a1")
        job.profiling_data = update_resource_profile(
            {},
            {
                "job_runtime(s)": 60.0,
                "max_process_memory(GB)": 80.0,
                "stage_generating_witness_max_memory(GB)": 6.0,
                "stage_proving_max_memory(GB)": 80.0,
            },
            setup_key="setup-a",
        )
        scheduler = JobScheduler("resource_aware")
        scheduler.add_jobs([job])
        node = NodeResources(
            node_id="node",
            memory_total_gb=64.0,
            memory_available_gb=20.0,
        )
        self.assertIsNone(scheduler.next_job(node))


class ResourceProfileTests(unittest.TestCase):
    def test_profile_keeps_bounded_history_and_uses_multiple_runs(self):
        profile = {}
        for memory_gb in (10.0, 12.0, 11.0):
            profile = update_resource_profile(
                profile,
                {
                    "job_runtime(s)": 30.0,
                    "max_process_memory(GB)": memory_gb,
                    "stage_proving_max_memory(GB)": memory_gb,
                    "stage_proving_avg_cpu_machine(%)": 50.0,
                },
                setup_key="setup-a",
            )
        estimates = get_profile_estimates(profile)
        self.assertEqual(estimates["samples"], 3)
        self.assertAlmostEqual(estimates["peak_memory_gb"], 12.0)
        self.assertAlmostEqual(estimates["stages"]["proving"]["memory_gb"], 12.0)


class StateStoreMigrationTests(unittest.TestCase):
    def test_old_worker_assignment_column_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE jobs (
                        job_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        assigned_worker_id TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                    ("job-1", "request-1", "IN_PROGRESS", "{}", "old-worker"),
                )
            CoordinatorStateStore(str(db_path))
            with sqlite3.connect(db_path) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                self.assertIn("assigned_node_id", columns)
                self.assertIn("lease_id", columns)
                self.assertIn("lease_updated_at", columns)
                node_id = conn.execute(
                    "SELECT assigned_node_id FROM jobs WHERE job_id = 'job-1'"
                ).fetchone()[0]
                self.assertEqual(node_id, "old-worker")


class StageAdmissionTests(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (
            ("psutil.virtual_memory", {"return_value": _memory()}),
            ("psutil.pid_exists", {"side_effect": _pid_exists}),
            ("process_tree_memory_gb", {"return_value": 0.0}),
        ):
            mocker = patch("zkinfer.runtime.node.stage_admission." + target, **kwargs)
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_live_memory_blocks_admission_despite_empty_reservation_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                tmp, "a", True, 32, memory_safety_reserve_gb=8
            )
            with patch("zkinfer.runtime.node.stage_admission.psutil.virtual_memory", side_effect=[
                _memory(64, 60), _memory(64, 60), _memory(64, 9), _memory(64, 60),
            ]), patch("zkinfer.runtime.node.stage_admission.time.sleep") as sleep:
                with controller.hold("PROVING"):
                    self.assertEqual(sleep.call_count, 1)

    def test_existing_rss_is_not_charged_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                tmp, "a", True, 20, memory_safety_reserve_gb=8
            )
            with patch("zkinfer.runtime.node.stage_admission.psutil.virtual_memory", return_value=_memory(64, 15)), \
                    patch("zkinfer.runtime.node.stage_admission.process_tree_memory_gb", return_value=14), \
                    patch("zkinfer.runtime.node.stage_admission.time.sleep", side_effect=AssertionError("unexpected wait")):
                with controller.hold("PROVING"):
                    pass

    def test_other_reservations_account_for_unallocated_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                tmp, "a", True, 10, memory_safety_reserve_gb=8
            )
            with patch.object(controller, "_read_reservations", return_value=[
                {"pid": os.getpid() + 1, "memory_gb": 20},
            ]), patch("zkinfer.runtime.node.stage_admission.psutil.virtual_memory", side_effect=[
                _memory(64, 60), _memory(64, 60), _memory(64, 29), _memory(64, 60),
            ]), patch("zkinfer.runtime.node.stage_admission.process_tree_memory_gb", return_value=4), \
                    patch("zkinfer.runtime.node.stage_admission.time.sleep") as sleep:
                with controller.hold("PROVING"):
                    self.assertEqual(sleep.call_count, 1)

    def test_unknown_job_can_start_with_os_memory_already_in_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                tmp, "a", True, 0, memory_safety_reserve_gb=8
            )
            with patch("zkinfer.runtime.node.stage_admission.process_tree_memory_gb", return_value=1), \
                    patch("zkinfer.runtime.node.stage_admission.time.sleep", side_effect=AssertionError("unexpected wait")):
                with controller.hold("CALIBRATING") as info:
                    self.assertAlmostEqual(info["memory_gb"], 53)

    def test_partial_fused_history_does_not_drop_unobserved_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = StageResourceHistory(Path(tmp) / "profiles")
            history.update_from_metrics("a", {"stage_key_gen_max_memory(GB)": 12})
            estimate = history.estimate("a", "FUSED_KEYGEN_PROVE", 40, .9)
            self.assertEqual(estimate["memory_gb"], 40)

    def test_resource_reservations_are_shared_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                root_dir=tmp,
                setup_key="setup-a",
                enabled=True,
                fallback_memory_gb=0.01,
                memory_safety_reserve_gb=0,
                poll_interval_s=0.02,
            )
            output: mp.Queue = mp.Queue()
            with controller.hold("PROVING"):
                child = mp.Process(
                    target=_wait_on_stage_admission,
                    args=(tmp, output),
                )
                child.start()
                time.sleep(0.15)
                self.assertTrue(output.empty())
            child.join(timeout=3)
            self.assertEqual(child.exitcode, 0)
            wait_s, elapsed_s = output.get(timeout=1)
            self.assertGreaterEqual(wait_s, 0.10)
            self.assertGreaterEqual(elapsed_s, 0.10)

    def test_historical_profile_replaces_fallback_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = StageResourceHistory(Path(tmp) / "profiles")
            history.update_from_metrics(
                "setup-a",
                {
                    "stage_proving_max_memory(GB)": 12.5,
                    "stage_proving_avg_cpu_machine(%)": 44.0,
                },
            )
            estimate = history.estimate(
                setup_key="setup-a",
                stage="PROVING",
                fallback_memory_gb=60.0,
                history_percentile=0.90,
            )
            self.assertEqual(estimate["source"], "historical")
            self.assertAlmostEqual(estimate["memory_gb"], 12.5)

    def test_admission_can_be_disabled_for_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = StageAdmissionController(
                root_dir=tmp,
                setup_key="setup-a",
                enabled=False,
                fallback_memory_gb=1.0,
                memory_safety_reserve_gb=0,
            )
            with controller.hold("PROVING") as info:
                self.assertFalse(info["enabled"])
                self.assertEqual(info["wait_time_s"], 0.0)


if __name__ == "__main__":
    unittest.main()

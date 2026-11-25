"""Resource profiles must not control whether proving artifacts are reused."""
import ast
import __future__
import importlib.util
import json
import logging
import os
import shutil
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_module("profile_test_config", "zkinfer/config/runtime.py")
profiles = load_module("profile_test_history", "zkinfer/runtime/resource_profiles.py")


def class_from_source(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class ProfileIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = config.SetupStoreConfig(False, "filesystem", str(self.root / "setup"), None, "",
                                             resource_profiles_root_dir=str(self.root / "profiles"))
        self.profile_dir = self.root / "profiles/model/partition"
        self.profile_dir.mkdir(parents=True)
        self.cold = {"setup_key": "key", "job_runtime(s)": 100, "max_process_memory(GB)": 40}
        self.warm = {"setup_key": "key", "job_runtime(s)": 20, "max_process_memory(GB)": 10}
        (self.profile_dir / "cold_resource_profile.json").write_text(json.dumps(self.cold))
        (self.profile_dir / "resource_profile.json").write_text(json.dumps(self.warm))
        ns = dict(logging=logging, os=os, time=time, Path=Path, EZKL_SETUP_IDENTITY={},
                  ProofJob=lambda **kw: SimpleNamespace(**kw),
                  get_profile_estimates=profiles.get_profile_estimates,
                  max_stage_memory_gb=profiles.max_stage_memory_gb,
                  exists=lambda path, **kw: Path(path).exists(),
                  load_json=lambda path, **kw: json.loads(Path(path).read_text()))
        Builder = class_from_source("zkinfer/runtime/request_builder.py", "RequestBuilder", ns)
        self.builder = Builder(logging.getLogger(__name__))
        self.builder._load_or_split_model = Mock(return_value=[SimpleNamespace(
            name="part", model=SimpleNamespace(SerializeToString=lambda: b"model"),
            sub_hash="hash", parent_hash="parent", input_data={})])
        self.shared = SimpleNamespace(
            setup_key_for_model=lambda *a: "key", put_model=lambda *a, **kw: "model-ref",
            partition_path=lambda *a: "model/partition",
            setup_ref=lambda key, **kw: SimpleNamespace(uri=str(self.root / "setup/key")),
            proof_prefix=lambda *a: "output-ref", put_input=lambda *a: "input-ref")
        self.request = SimpleNamespace(name="model", request_id="req", preparation_metrics={}, split_mode="none", onnx_model_path="model.onnx")

    def jobs(self, store=None):
        return self.builder.build_jobs(self.request, self.shared, store or self.store, 1)

    def test_cold_loads_memory_and_duration_without_reuse(self):
        job = self.jobs()[0]
        self.assertFalse(self.store.enabled)
        self.assertEqual(job.predicted_peak_memory_gb, 40)
        self.assertEqual(job.predicted_duration_s, 100)
        self.assertEqual(job.profiling_file_path, str(self.profile_dir / "cold_resource_profile.json"))

    def test_warm_still_loads_existing_filename(self):
        from dataclasses import replace
        job = self.jobs(replace(self.store, enabled=True))[0]
        self.assertEqual(job.predicted_peak_memory_gb, 10)
        self.assertEqual(job.predicted_duration_s, 20)

    def test_missing_cold_profile_never_falls_back_to_warm(self):
        (self.profile_dir / "cold_resource_profile.json").unlink()
        job = self.jobs()[0]
        self.assertEqual(job.predicted_peak_memory_gb, 0)
        self.assertEqual(job.profiling_data, {})
        self.assertIsNotNone(job.profiling_file_path)  # Later completion can save history.

    def test_profiles_can_be_disabled_independently(self):
        from dataclasses import replace
        store = replace(self.store, enabled=True, resource_profiles_enabled=False)
        job = self.jobs(store)[0]
        self.assertTrue(store.enabled)
        self.assertIsNone(job.profiling_file_path)
        self.assertEqual(job.predicted_peak_memory_gb, 0)

    def test_config_defaults_and_separate_s3_profile_prefix(self):
        cfg = OmegaConf.create({"storage": {"backend": "filesystem", "reusable_setup": {
            "enabled": False, "path": str(self.root / "setup")}}})
        store = config._resolve_setup_store_config(cfg)
        self.assertFalse(store.enabled)
        self.assertTrue(store.resource_profiles_enabled)
        self.assertEqual(store.resource_profiles_root_dir, store.root_dir)
        cfg.storage.backend = "s3"
        cfg.storage.s3_bucket = "test"
        cfg.storage.s3_prefix = "prefix"
        cfg.storage.resource_profiles = {"enabled": True, "path": "profiles"}
        store = config._resolve_setup_store_config(cfg)
        self.assertEqual(store.resource_profiles_root_dir, "prefix/profiles")
        self.assertFalse(store.enabled)

    def test_cold_profile_saved_and_next_request_uses_it(self):
        # Exercise the real coordinator's persistence method, substituting only
        # report rendering (unrelated CSV/statistics) with the completed report.
        source = ast.parse((ROOT / "zkinfer/runtime/coordinator.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Coordinator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_write_job_report_and_profile")
        report = {"setup_key": "key", "job_runtime(s)": 120, "max_process_memory(GB)": 42}
        ns = dict(write_job_report=lambda *a, **kw: report,
                  save_json=lambda obj, path, **kw: Path(path).write_text(json.dumps(obj)),
                  get_profile_estimates=profiles.get_profile_estimates,
                  max_stage_memory_gb=profiles.max_stage_memory_gb,
                  JobStatus=SimpleNamespace(COMPLETED="done", QUEUED="queued"))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "coordinator", "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        job = self.jobs()[0]
        job.artifact_manifest_ref = None
        job.job_status = "done"
        owner = SimpleNamespace(setup_store=self.store, lock=threading.RLock(),
                                active_requests={}, state_store=None)
        ns["_write_job_report_and_profile"](owner, job, SimpleNamespace(reports_dir="unused"), {})
        self.assertEqual(self.jobs()[0].predicted_peak_memory_gb, 42)
        self.assertEqual(json.loads((self.profile_dir / "resource_profile.json").read_text()), self.warm)

    def test_cold_worker_uses_empty_private_setup_not_cached_keys(self):
        # Run actual worker task preparation with storage transports substituted.
        tree = ast.parse((ROOT / "zkinfer/runtime/node/supervisor.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NodeSupervisor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "prepare_local_task")
        transfer = SimpleNamespace(
            materialize=lambda ref, name: (name, {"materialize_time(s)": 0.0}),
            restore_setup=Mock(side_effect=AssertionError("cold must not restore setup")),
        )
        ns = dict(time=time, shutil=shutil, Path=Path, copy_run_metadata=lambda *a: None,
                  from_pb_artifact=lambda value: value,
                  ArtifactTransfer=lambda workspace: transfer,
                  LocalProverTask=lambda **kw: SimpleNamespace(**kw))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "supervisor", "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        node = SimpleNamespace(resource_sampling_interval_sec=0.25)
        cfg = SimpleNamespace(setup_store=self.store, node=node,
                              request_store=SimpleNamespace(backend="filesystem", s3_bucket=None))
        owner = SimpleNamespace(
            cfg=cfg, node_id="node", metadata_snapshot=None,
            stage_admission_dir=str(self.root / "state"),
            stage_cfg=SimpleNamespace(enabled=True, mode="whole_job", memory_safety_reserve_gb=8,
                                      history_percentile=0.9),
            _job_paths=lambda lease: (self.root / "request", self.root / "tmp", self.root / "out"))
        lease = SimpleNamespace(lease_id="lease", setup_key="key", job_id="job", request_id="req",
                                model=None, input=SimpleNamespace(uri="input.json"), predicted_peak_memory_gb=40,
                                output_prefix=SimpleNamespace(uri="output", backend="filesystem", s3_bucket=""))
        task = ns["prepare_local_task"](owner, lease)
        self.assertEqual(Path(task.setup_dir), self.root / "tmp/setup")
        self.assertEqual(list(Path(task.setup_dir).iterdir()), [])
        self.assertEqual(task.predicted_peak_memory_gb, 40)
        transfer.restore_setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()

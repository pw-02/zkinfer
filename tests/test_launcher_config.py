"""Tests for experiment config translation and descriptive run names."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from experiments.scripts.launcher_config import (
    CONFIG_DIR,
    configure_run_paths,
    load_config,
    make_run_name,
)


class LauncherConfigTests(unittest.TestCase):
    def setUp(self):
        self.user, self.cfg = load_config(CONFIG_DIR / "config.yaml", workload="bert")
        self.now = datetime(2026, 11, 25, 12, 30, 0)

    def test_effective_workload_and_name(self):
        self.assertEqual(self.user.workload, "bert")
        self.assertEqual(list(self.cfg.suite.workloads), ["bert"])
        self.assertEqual(self.cfg.preparation.dispatch, "streaming")
        self.assertTrue(self.cfg.cache.circuits)
        self.user.partitioning.enabled = True
        self.user.partitioning.size = 4
        self.user.cache["keys"] = False
        self.user.scheduling.policy = "fifo"
        self.user.scheduling.memory_admission = "disabled"
        self.user.scheduling.max_concurrent_jobs = 1
        self.assertEqual(
            make_run_name(self.user, now=self.now, suffix="abcd"),
            "2026-11-25_12-30-00_bert_p4_c1_fifo_abcd",
        )

    def test_internal_run_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            configure_run_paths(self.cfg, self.user, root)
            self.assertEqual(self.cfg.storage.request_path, str(root / ".runtime" / "storage" / "requests"))
            self.assertEqual(self.cfg.coordinator.state_db_path, str(root / ".runtime" / "coordinator_state.sqlite3"))
            self.assertEqual(self.cfg.node.runs_dir, str(root / "requests"))

    def test_monolithic_and_auto(self):
        self.user.partitioning.enabled = False
        self.user.scheduling.max_concurrent_jobs = None
        self.user.scheduling.memory_admission = "stage"
        self.assertIn("_mono_", make_run_name(self.user, now=self.now))
        self.assertIn("_cauto_", make_run_name(self.user, now=self.now))

    def test_history_flags_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            for use in (False, True):
                for update in (False, True):
                    self.user.resource_model.use_history = use
                    self.user.resource_model.update_history = update
                    OmegaConf.save(self.user, path)
                    _, cfg = load_config(path, run_dir=tmp)
                    self.assertEqual(cfg.resource_model.use_history, use)
                    self.assertEqual(cfg.resource_model.update_history, update)
                    self.assertEqual(cfg.cache.path, "storage/cache")

    def test_independent_rayon_settings(self):
        from experiments.run_local_exp import common_overrides
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            self.assertIsNone(self.cfg.resources.coordinator_threads)
            self.assertIsNone(self.cfg.resources.prover_threads)
            self.user.resources.coordinator_threads = 4
            self.user.resources.prover_threads = 16
            OmegaConf.save(self.user, path)
            _, cfg = load_config(path, run_dir=tmp)
            self.assertEqual(cfg.resources.coordinator_threads, 4)
            self.assertEqual(cfg.resources.prover_threads, 16)
            self.assertIn("resources.coordinator_threads=4", common_overrides(cfg))
            self.assertIn("resources.prover_threads=16", common_overrides(cfg))
            for value in (0, -1, True, 1.5, "eight"):
                self.user.resources.prover_threads = value
                OmegaConf.save(self.user, path)
                with self.assertRaisesRegex(ValueError, "resources.prover_threads"):
                    load_config(path)

    def test_key_representation(self):
        from experiments.run_local_exp import common_overrides
        from zkinfer.config.runtime import build_runtime_config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            for value in ("full", "reduced", "compact"):
                self.user.proving.key_representation = value
                OmegaConf.save(self.user, path)
                _, cfg = load_config(path, run_dir=tmp)
                self.assertEqual(build_runtime_config(cfg).node.key_representation, value)
                self.assertIn(f"proving.key_representation={value}", common_overrides(cfg))
            self.user.proving.key_representation = "invalid"
            OmegaConf.save(self.user, path)
            with self.assertRaisesRegex(ValueError, "proving.key_representation"):
                load_config(path)

    def test_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            self.user.scheduling.max_concurrent_jobs = None
            self.user.scheduling.memory_admission = "disabled"
            OmegaConf.save(self.user, path)
            with self.assertRaisesRegex(ValueError, "requires memory admission"):
                load_config(path)
            with self.assertRaisesRegex(ValueError, "Unknown workload"):
                load_config(path, workload="not_a_real_workload")


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(self.cfg.coordinator.witness_preparation, "upfront")
        self.assertTrue(self.cfg.verification.enabled)
        self.assertEqual(self.cfg.coordinator.preparation_dispatch, "streaming")
        self.assertTrue(self.cfg.coordinator.preparation_cache_enabled)
        self.user.partitioning.enabled = True
        self.user.partitioning.size = 4
        self.user.cache.reuse.keys = False
        self.user.scheduling.policy = "fifo"
        self.user.scheduling.memory_admission = "disabled"
        self.user.scheduling.max_concurrent_jobs = 1
        self.assertEqual(
            make_run_name(self.user, now=self.now, suffix="abcd"),
            "2026-11-25_12-30-00_bert_p4_keys-off_c1_fifo_noadm_exact_streaming_circuits-on_profiles-on_persistent_abcd",
        )

    def test_monolithic_and_auto(self):
        self.user.partitioning.enabled = False
        self.user.scheduling.max_concurrent_jobs = None
        self.user.scheduling.memory_admission = "stage"
        self.assertIn("_mono_", make_run_name(self.user, now=self.now))
        self.assertIn("_cauto_", make_run_name(self.user, now=self.now))

    def test_run_owned_cache_and_profile_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.user.cache.scope = "run"
            self.user.cache.reuse.profiles = False
            configure_run_paths(self.cfg, self.user, root)
            self.assertEqual(self.cfg.storage.reusable_setup.path, str(root / "storage/cache/setup"))
            self.assertEqual(self.cfg.node.runtime_state_dir, str(root / "storage/runtime"))
            self.assertEqual(self.cfg.storage.request_data.path, str(root / "storage/requests"))

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

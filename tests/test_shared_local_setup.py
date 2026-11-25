"""Regression coverage for private job snapshots and shared artifact transfers."""
import hashlib
import shutil
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.artifact_transfer import ArtifactTransfer
from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages


class ArtifactTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.ref = ArtifactRef("filesystem", str(self.shared))
        self.transfer = ArtifactTransfer(self.root / "transfer")
        self.source = self.root / "source"
        self.source.mkdir()
        for name in ArtifactTransfer.SETUP_FILES:
            (self.source / name).write_bytes(b'{"run_args":{"logrows":1}}')
        self.transfer.publish_setup(self.ref, "key", self.source, False)

    def test_materialization_validates_hash_and_keeps_private_copy(self):
        source = self.source / "settings.json"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        ref = ArtifactRef("filesystem", str(source), sha256=digest)
        path, metrics = self.transfer.materialize(ref, "settings.json")
        self.assertEqual(Path(path).read_bytes(), source.read_bytes())
        self.assertNotIn("cache_hit", metrics)
        source.write_bytes(b"changed")
        self.assertNotEqual(Path(path).read_bytes(), source.read_bytes())
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.transfer.materialize(ref, "bad.json")

    def test_missing_artifact_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.transfer.materialize(ArtifactRef("filesystem", str(self.root / "missing")), "missing")

    def test_snapshot_survives_replacement_and_cleanup(self):
        first = self.root / "first"
        self.transfer.restore_setup(self.ref, "key", first, False)
        original = (first / "pk.json").read_bytes()
        for name in ArtifactTransfer.SETUP_FILES:
            (self.source / name).write_bytes(b"new")
        self.transfer.publish_setup(self.ref, "new-key", self.source, False)
        self.assertEqual((first / "pk.json").read_bytes(), original)
        self.assertEqual((self.shared / "pk.json").read_bytes(), b"new")
        shutil.rmtree(first)
        self.assertTrue((self.shared / "pk.json").exists())

    def test_concurrent_readers_have_separate_snapshots(self):
        def restore(i):
            target = self.root / f"reader-{i}"
            self.transfer.restore_setup(self.ref, "key", target, False)
            return (target / "pk.json").read_bytes()
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(len(set(pool.map(restore, range(12)))), 1)

    def test_incompatible_or_incomplete_bundle_is_not_restored(self):
        target = self.root / "incompatible"
        self.transfer.restore_setup(self.ref, "other-key", target, False)
        self.assertEqual(list(target.iterdir()), [])
        (self.shared / "pk.json").unlink()
        target = self.root / "incomplete"
        self.transfer.restore_setup(self.ref, "key", target, False)
        self.assertEqual(list(target.iterdir()), [])

    def test_prepared_keys_do_not_overwrite_circuit(self):
        for name in ArtifactTransfer.KEY_FILES:
            (self.source / name).write_bytes(b"new key")
        original = (self.shared / "settings.json").read_bytes()
        self.transfer.publish_setup(self.ref, "prepared", self.source, True)
        target = self.root / "prepared"
        self.transfer.restore_setup(self.ref, "prepared", target, True)
        self.assertEqual(set(p.name for p in target.iterdir()), set(ArtifactTransfer.KEY_FILES))
        self.assertEqual((self.shared / "settings.json").read_bytes(), original)

    def test_s3_materializes_into_private_workspace(self):
        def download(bucket, key, destination):
            Path(destination).write_bytes(b"downloaded")
        with patch("zkinfer.runtime.node.artifact_transfer.s3.download_file", side_effect=download):
            path, metrics = self.transfer.materialize(
                ArtifactRef("s3", "model.onnx", s3_bucket="bucket",
                            sha256=hashlib.sha256(b"downloaded").hexdigest()), "model.onnx")
        self.assertEqual(Path(path).read_bytes(), b"downloaded")
        self.assertTrue(Path(path).is_relative_to(self.transfer.workspace))
        self.assertEqual(metrics["bytes_downloaded"], len(b"downloaded"))

    def test_read_only_prover_rejects_missing_keys_and_writes(self):
        with patch.dict(sys.modules, {"ezkl": types.ModuleType("ezkl")}):
            prover = EZKLProofStages(
                input_data_path="input", onnx_model_path="model",
                artifact_dir=str(self.root / "output"),
                proving_cache_enabled=True, proving_cache_overwrite=False,
                proving_cache_type="filesystem", proving_cache_root_dir=str(self.shared),
                proving_cache_read_only=True, local_tmp_dir=str(self.root / "tmp"),
                srs_dir=str(self.root / "srs"),
            )
        self.assertEqual(prover._try_load_keys_from_cache(), (True, 0.0))
        self.assertTrue(prover.calibrate_settings_with_fallback()[0])
        self.assertTrue(prover.compile_circuit()[0])
        with self.assertRaises(RuntimeError):
            prover._save_settings({})
        with self.assertRaises(RuntimeError):
            prover.setup_and_compute_proof()
        (self.shared / "pk.json").unlink()
        with self.assertRaises(FileNotFoundError):
            prover._try_load_keys_from_cache()
        self.assertFalse((self.shared / "pk.json").exists())


if __name__ == "__main__":
    unittest.main()

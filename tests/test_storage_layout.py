from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

import zkinfer.proto.zkservice_pb2 as pb
from experiments.run_local_exp import cleanup_request_store, common_overrides
from experiments.scripts.launcher_config import load_config
from tests.test_serving_reliability import runtime_config
from zkinfer.config.runtime import build_runtime_config
from zkinfer.runtime.models import InferenceRequest
from zkinfer.runtime.node.artifact_transfer import ArtifactTransfer
from zkinfer.runtime.node.supervisor import NodeSupervisor, to_pb_artifact
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.storage.shared_store import SharedStorage


def build_jobs(cfg, shared, name="model"):
    request = InferenceRequest(name, "model.onnx", "input.json", "none", 1, "fifo")
    partition = SimpleNamespace(
        name="partition", parent_hash=hashlib.sha256(b"parent").hexdigest(), sub_hash=hashlib.sha256(b"fixed-model").hexdigest(),
        model=SimpleNamespace(SerializeToString=lambda: b"fixed-model"),
        input_data={"input_data": [[1, 2]]},
    )
    builder = RequestBuilder(logging.getLogger("test-storage"))
    with patch.object(builder, "_load_or_split_model", return_value=[partition]):
        jobs = builder.build_jobs(request, shared, cfg.setup_store, 1)
    return request, jobs[0]


def lease_for(job, lease_id="attempt"):
    return pb.JobLease(
        job_available=True, job_id=job.job_id, request_id=job.inference_request_id,
        lease_id=lease_id, setup_key=job.artifact_key,
        model=to_pb_artifact(job.model_ref), input=to_pb_artifact(job.input_ref),
        setup_cache=to_pb_artifact(job.setup_cache_ref), output_prefix=to_pb_artifact(job.output_prefix_ref),
    )


class SharedStorageTests(unittest.TestCase):
    def test_model_bundle_is_reused_but_inputs_and_outputs_belong_to_each_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = runtime_config(root)
            shared = SharedStorage("filesystem", cfg.setup_store.root_dir, cfg.request_store.root_dir)
            first_request, first = build_jobs(cfg, shared, "first-name")
            _, second = build_jobs(cfg, shared, "second-name")
            key = hashlib.sha256(b"fixed-model").hexdigest()
            self.assertEqual(first.model_ref.sha256, key)
            key = first.artifact_key
            namespace = shared.partition_path("model", hashlib.sha256(b"parent").hexdigest(), b"fixed-model", "monolithic")
            self.assertEqual(first.model_ref.uri, str(root / "setup" / namespace / "model.onnx"))
            self.assertEqual(first.setup_cache_ref.uri, str(root / "setup" / namespace))
            self.assertEqual(first.model_ref, second.model_ref)
            self.assertEqual(first.setup_cache_ref, second.setup_cache_ref)
            self.assertEqual(Path(first.model_ref.uri).read_bytes(), b"fixed-model")
            self.assertNotEqual(first.input_ref.uri, second.input_ref.uri)
            self.assertNotEqual(first.output_prefix_ref.uri, second.output_prefix_ref.uri)
            self.assertEqual(Path(first.input_ref.uri).parent, root / "requests-store" / first_request.request_id / "inputs")
            self.assertEqual(Path(first.output_prefix_ref.uri).parent, root / "requests-store" / first_request.request_id / "proofs")
            self.assertEqual(json.loads(Path(first.input_ref.uri).read_text()), {"input_data": [[1, 2]]})

    def test_short_namespace_rejects_full_parent_hash_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = SharedStorage("filesystem", str(root / "cache"), str(root / "requests"))
            first = "a" * 16 + "b" * 48
            second = "a" * 16 + "c" * 48
            namespace = shared.partition_path("model", first, b"partition", "matmul")
            self.assertEqual(namespace.split("/")[0], "model_" + first[:16])
            self.assertEqual(
                (root / "cache" / namespace / ".identity.sha256").read_text().strip(),
                hashlib.sha256(b"partition").hexdigest(),
            )
            self.assertEqual(shared.partition_path("model", first, b"partition", "matmul"), namespace)
            with self.assertRaisesRegex(ValueError, "different content"):
                shared.partition_path("model", second, b"other", "matmul")

    def test_request_input_cannot_be_overwritten_with_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = SharedStorage("filesystem", str(Path(tmp) / "setup"), str(Path(tmp) / "requests"))
            ref = shared.put_input("request", "part", {"value": 1})
            with self.assertRaisesRegex(ValueError, "different content"):
                shared.put_input("request", "part", {"value": 2})
            self.assertEqual(json.loads(Path(ref.uri).read_text()), {"value": 1})

    def test_s3_prefixes_follow_same_layout_and_model_upload_is_deduplicated(self):
        objects = {}
        with patch("zkinfer.storage.shared_store.s3.exists", side_effect=lambda bucket, key: key in objects), \
                patch("zkinfer.storage.shared_store.s3.put_bytes", side_effect=lambda data, bucket, key: objects.setdefault(key, data)) as put, \
                patch("zkinfer.storage.shared_store.s3.get_bytes", side_effect=lambda bucket, key: objects[key]):
            shared = SharedStorage("s3", "prefix/setup", "prefix/requests", "bucket")
            model = shared.put_model(b"model")
            self.assertEqual(shared.put_model(b"model"), model)
            self.assertEqual(put.call_count, 1)
            self.assertEqual(model.uri, "prefix/setup/" + model.sha256 + "/model.onnx")
            input_ref = shared.put_input("r", "p", {"x": 1})
            self.assertEqual(input_ref.uri, "prefix/requests/r/inputs/p.json")
            self.assertEqual(shared.proof_prefix("r", "p").uri, "prefix/requests/r/proofs/p")
            with self.assertRaises(ValueError):
                shared.put_input("r", "p", {"x": 2})
            self.assertEqual(input_ref.s3_bucket, "bucket")

    def test_path_components_cannot_escape_their_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = SharedStorage("filesystem", str(Path(tmp) / "setup"), str(Path(tmp) / "requests"))
            for component in ("", "..", "../outside", "a/b", "a\\b"):
                with self.subTest(component=component), self.assertRaises(ValueError):
                    shared.put_input(component, "part", {})


class NodeSetupLayoutTests(unittest.TestCase):
    def test_second_node_restores_bundle_and_cold_mode_ignores_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = runtime_config(root)
            shared = SharedStorage("filesystem", cfg.setup_store.root_dir, cfg.request_store.root_dir)
            _, job = build_jobs(cfg, shared)
            first = NodeSupervisor(cfg)
            task = first.prepare_local_task(lease_for(job, "first"))
            self.assertEqual(task.model_path, job.model_ref.uri)
            self.assertFalse((Path(task.setup_dir) / "pk.json").exists())
            for filename in ArtifactTransfer.SETUP_FILES:
                (Path(task.setup_dir) / filename).write_bytes(b"derived setup")
            ArtifactTransfer(Path(task.tmp_dir) / "transfer").publish_setup(
                job.setup_cache_ref, job.artifact_key, task.setup_dir, False)
            bundle = Path(job.setup_cache_ref.uri)
            self.assertTrue((bundle / "model.onnx").exists())
            self.assertTrue((bundle / "pk.json").exists())
            self.assertTrue((bundle / "metadata.json").exists())
            second_cfg = replace(cfg, node=replace(cfg.node,
                runs_dir=str(root / "second-runs"), runtime_state_dir=str(root / "second-runtime")))
            second = NodeSupervisor(second_cfg)
            warm = second.prepare_local_task(lease_for(job, "warm"))
            self.assertNotEqual(warm.setup_dir, task.setup_dir)
            self.assertEqual(Path(warm.cached_pk_path), bundle / "pk.json")
            self.assertEqual(warm.transfer_metrics["model"]["bytes_downloaded"], 0)
            self.assertGreater(warm.transfer_metrics["setup"]["bytes_referenced"], 0)
            cold_node = NodeSupervisor(replace(second_cfg, setup_store=replace(cfg.setup_store, enabled=False)))
            cold = cold_node.prepare_local_task(lease_for(job, "cold"))
            self.assertFalse((Path(cold.setup_dir) / "pk.json").exists())
            self.assertEqual(cold.transfer_metrics["setup"]["bytes_downloaded"], 0)
            self.assertNotIn("cache_hit", cold.transfer_metrics["model"])
            self.assertEqual((bundle / "pk.json").read_bytes(), b"derived setup")

    def test_cold_request_still_has_reusable_model_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = runtime_config(Path(tmp))
            shared = SharedStorage("filesystem", cfg.setup_store.root_dir, cfg.request_store.root_dir)
            cold_cfg = replace(cfg, setup_store=replace(cfg.setup_store, enabled=False))
            _, cold = build_jobs(cold_cfg, shared)
            self.assertFalse(cold.setup_cache_ref.is_empty)
            self.assertTrue(Path(cold.model_ref.uri).exists())
            self.assertTrue(cold.profiling_file_path.endswith("cold_resource_profile.json"))


class StorageConfigurationTests(unittest.TestCase):
    def test_default_layout_and_legacy_backend_rejection(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
        runtime = build_runtime_config(cfg)
        self.assertEqual(Path(runtime.setup_store.root_dir).name, "cache")
        self.assertEqual(Path(runtime.request_store.root_dir).name, "requests")
        cfg.storage.backend = "s3"
        with self.assertRaisesRegex(ValueError, "filesystem"):
            build_runtime_config(cfg)

    def test_overlapping_roots_are_rejected(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
        cfg.storage.request_path = "storage"
        with self.assertRaisesRegex(ValueError, "must be separate"):
            build_runtime_config(cfg)

    def test_suite_cleanup_deletes_only_its_requests_and_preserves_setup_and_other_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = OmegaConf.create({
                "storage": {"request_path": str(root / "requests")},
                "cache": {"path": str(root / "setup"), "keys": True},
                "output": {"keep_proofs": False},
                "paths": {"runs_dir": str(root / "runs")},
            })
            own = root / "requests/own/proofs"
            other = root / "requests/other/proofs"
            model = root / "setup/key/model.onnx"
            for path in (own, other, model):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            cleanup_request_store(cfg, Mock(), ["own"])
            self.assertFalse(own.exists())
            self.assertTrue(other.exists())
            self.assertTrue(model.exists())
            cfg.output.keep_proofs = True
            cleanup_request_store(cfg, Mock(), ["other"])
            self.assertTrue(other.exists())

    def test_experiment_overrides_use_current_runtime_keys(self):
        _, cfg = load_config(Path(__file__).parents[1] / "experiments/config/config.yaml")
        cfg.cache.keys = False
        overrides = common_overrides(cfg)
        self.assertIn("cache.path=storage/cache", overrides)
        self.assertIn("storage.request_path=storage/requests", overrides)
        self.assertIn("cache.keys=false", overrides)
        self.assertIn(f"scheduling.policy={cfg.scheduling.policy}", overrides)
        with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "zkinfer/config"), version_base=None):
            resolved = compose(config_name="config", overrides=overrides)
        runtime = build_runtime_config(resolved)
        self.assertFalse(runtime.setup_store.enabled)
        self.assertEqual(Path(runtime.request_store.root_dir).name, "requests")


if __name__ == "__main__":
    unittest.main()

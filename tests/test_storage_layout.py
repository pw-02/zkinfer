from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

import zkinfer.proto.zkservice_pb2 as pb
from experiments.run_local_exp import cleanup_request_store, common_overrides
from tests.test_serving_reliability import runtime_config
from zkinfer.config.runtime import build_runtime_config
from zkinfer.runtime.models import InferenceRequest
from zkinfer.runtime.node.cache import NodeCache
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
            self.assertEqual(first.setup_cache_ref.uri, str(root / "setup" / namespace / key))
            self.assertEqual(first.model_ref, second.model_ref)
            self.assertEqual(first.setup_cache_ref, second.setup_cache_ref)
            self.assertEqual(Path(first.model_ref.uri).read_bytes(), b"fixed-model")
            self.assertNotEqual(first.input_ref.uri, second.input_ref.uri)
            self.assertNotEqual(first.output_prefix_ref.uri, second.output_prefix_ref.uri)
            self.assertEqual(Path(first.input_ref.uri).parent, root / "requests-store" / first_request.request_id / "inputs")
            self.assertEqual(Path(first.output_prefix_ref.uri).parent, root / "requests-store" / first_request.request_id / "proofs")
            self.assertEqual(json.loads(Path(first.input_ref.uri).read_text()), {"input_data": [[1, 2]]})

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
    def test_second_node_restores_bundle_and_cold_mode_ignores_resident_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = runtime_config(root)
            shared = SharedStorage("filesystem", cfg.setup_store.root_dir, cfg.request_store.root_dir)
            _, job = build_jobs(cfg, shared)
            first_node = NodeSupervisor(cfg)
            with ExitStack() as pins:
                task = first_node.prepare_local_task(lease_for(job, "first"), pins)
                self.assertEqual(Path(task.model_path).parent, Path(task.setup_dir))
                self.assertFalse(first_node.cache.setup_ready(job.artifact_key))
                self.assertEqual(first_node.cache.cached_setup_keys(), [])
                for filename in NodeCache.SETUP_FILES:
                    (Path(task.setup_dir) / filename).write_bytes(b"derived setup")
                first_node.cache.publish_setup_with_metrics(task.setup_dir, job.setup_cache_ref)
            bundle = Path(job.setup_cache_ref.uri)
            self.assertTrue((bundle / "model.onnx").exists())
            self.assertTrue((bundle / "pk.json.gz").exists())
            second_cfg = replace(cfg, node=replace(cfg.node,
                local_cache_dir=str(root / "second-cache"), runtime_state_dir=str(root / "second-runtime")))
            second_node = NodeSupervisor(second_cfg)
            with ExitStack() as pins:
                warm = second_node.prepare_local_task(lease_for(job, "warm"), pins)
                self.assertEqual(Path(warm.model_path).parent, Path(warm.setup_dir))
                self.assertTrue(second_node.cache.setup_ready(job.artifact_key))
                self.assertEqual((Path(warm.setup_dir) / "pk.json").read_bytes(), b"derived setup")
                self.assertEqual(warm.transfer_metrics["model"]["bytes_downloaded"], len(b"fixed-model"))
                self.assertGreater(warm.transfer_metrics["setup"]["bytes_downloaded"], 0)
            cold_node = NodeSupervisor(replace(second_cfg, setup_store=replace(cfg.setup_store, enabled=False)))
            with ExitStack() as pins:
                cold = cold_node.prepare_local_task(lease_for(job, "cold"), pins)
                self.assertEqual(cold.model_path, warm.model_path)
                self.assertNotEqual(cold.setup_dir, warm.setup_dir)
                self.assertFalse((Path(cold.setup_dir) / "pk.json").exists())
                self.assertEqual(cold.transfer_metrics["setup"]["bytes_downloaded"], 0)
                self.assertTrue(cold.transfer_metrics["model"]["cache_hit"])

    def test_cold_request_still_has_reusable_model_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = runtime_config(Path(tmp))
            shared = SharedStorage("filesystem", cfg.setup_store.root_dir, cfg.request_store.root_dir)
            cold_cfg = replace(cfg, setup_store=replace(cfg.setup_store, enabled=False))
            _, cold = build_jobs(cold_cfg, shared)
            self.assertFalse(cold.setup_cache_ref.is_empty)
            self.assertTrue(Path(cold.model_ref.uri).exists())
            self.assertIsNone(cold.profiling_file_path)


class StorageConfigurationTests(unittest.TestCase):
    def test_default_layout_and_s3_roots(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
        runtime = build_runtime_config(cfg)
        self.assertEqual(Path(runtime.setup_store.root_dir).name, "setup")
        self.assertEqual(Path(runtime.request_store.root_dir).name, "requests")
        cfg.storage.backend = "s3"
        cfg.storage.s3_bucket = "bucket"
        cfg.storage.s3_prefix = "prefix"
        cfg.storage.reusable_setup.path = "setup"
        cfg.storage.request_data.path = "requests"
        runtime = build_runtime_config(cfg)
        self.assertEqual(runtime.setup_store.root_dir, "prefix/setup")
        self.assertEqual(runtime.request_store.root_dir, "prefix/requests")
        cfg.storage.s3_prefix = ""
        runtime = build_runtime_config(cfg)
        self.assertEqual(runtime.setup_store.root_dir, "setup")
        self.assertEqual(runtime.request_store.root_dir, "requests")

    def test_overlapping_roots_are_rejected(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "zkinfer/config/config.yaml")
        cfg.storage.request_data.path = "storage"
        with self.assertRaisesRegex(ValueError, "must be separate"):
            build_runtime_config(cfg)
        cfg.storage.backend = "s3"
        cfg.storage.s3_bucket = "bucket"
        with self.assertRaisesRegex(ValueError, "must be separate"):
            build_runtime_config(cfg)

    def test_suite_cleanup_deletes_only_its_requests_and_preserves_setup_and_other_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = OmegaConf.create({
                "storage": {
                    "request_data": {"path": str(root / "requests")},
                    "reusable_setup": {"path": str(root / "setup"), "enabled": True},
                },
                "output": {"keep_request_data": False},
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
            cfg.output.keep_request_data = True
            cleanup_request_store(cfg, Mock(), ["other"])
            self.assertTrue(other.exists())

    def test_experiment_overrides_use_current_runtime_keys(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "experiments/config/config.yaml")
        overrides = common_overrides(cfg)
        self.assertIn("storage.reusable_setup.path=storage/setup", overrides)
        self.assertIn("storage.request_data.path=storage/requests", overrides)
        self.assertIn("storage.reusable_setup.enabled=false", overrides)
        self.assertIn("coordinator.scheduling_policy=lpt", overrides)
        with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "zkinfer/config"), version_base=None):
            resolved = compose(config_name="config", overrides=overrides)
        runtime = build_runtime_config(resolved)
        self.assertFalse(runtime.setup_store.enabled)
        self.assertEqual(Path(runtime.request_store.root_dir).name, "requests")


if __name__ == "__main__":
    unittest.main()

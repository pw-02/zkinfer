"""Test the parent IPC protocol with real short-lived subprocesses, no EZKL."""
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from zkinfer.config.runtime import SetupStoreConfig
from zkinfer.runtime.models import InferenceRequest, ProofJob
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.preparation_runner import run_preparation
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.runtime.state_store import CoordinatorStateStore
from zkinfer.storage.shared_store import SharedStorage


class PreparationProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = InferenceRequest("model", "model.onnx", "input.json", "none", 1, "fifo")
        self.request.run_dir = str(self.root / "run")
        self.builder = RequestBuilder(logging.getLogger("test-preparation"))
        self.shared = SharedStorage("filesystem", str(self.root / "cache"), str(self.root / "requests"))
        self.store = SetupStoreConfig(False, "filesystem", str(self.root / "cache"), None, "")
        ref = ArtifactRef.empty()
        self.job = ProofJob("model", "part", self.request.request_id, ref, ref, ref, ref)
        self.real_popen = subprocess.Popen

    def fake_child(self, events, wait_for_ack=False, code=0):
        def launch(args, **kwargs):
            source = """import json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
events = json.loads(sys.argv[2])
for i, event in enumerate(events):
    temp = root / '.tmp'
    temp.write_text(json.dumps(event))
    os.replace(temp, root / f'{i:08d}.json')
    if sys.argv[3] == 'True' and event['kind'] == 'job':
        deadline = time.monotonic() + 10
        while not (root.parent / 'ack').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('parent did not publish job before child exit')
            time.sleep(0.02)
sys.exit(int(sys.argv[4]))
"""
            return self.real_popen([sys.executable, "-c", source, args[-1],
                                    json.dumps(events), str(wait_for_ack), str(code)], **kwargs)
        return launch

    def run_child(self, **kwargs):
        return run_preparation(self.request, self.builder, self.shared,
                               self.store, 1, None, **kwargs)

    def job_event(self):
        return {"kind": "job", "job": CoordinatorStateStore._job_to_dict(self.job)}

    def test_job_published_before_process_exits(self):
        events = [{"kind": "plan", "count": 1}, self.job_event(),
                  {"kind": "complete", "metrics": {"child_metric": 12}}]
        published = []
        def publish(job):
            published.append(job.job_id)
            folder = next(Path(self.request.run_dir).glob(".preparation-*"))
            (folder / "ack").touch()
        with patch("zkinfer.runtime.preparation_runner.subprocess.Popen",
                   side_effect=self.fake_child(events, wait_for_ack=True)):
            jobs = self.run_child(on_job_ready=publish)
        self.assertEqual(published, [self.job.job_id])
        self.assertEqual(jobs[0].job_id, self.job.job_id)
        self.assertEqual(self.request.preparation_metrics["child_metric"], 12)
        self.assertTrue((Path(self.request.run_dir) / "request_preparation.log").exists())
        self.assertEqual(list(Path(self.request.run_dir).glob(".preparation-*")), [])

    def test_progress_event_is_logged(self):
        events = [{"kind": "plan", "count": 1},
                  {"kind": "progress", "message": "Calibrating circuit"},
                  self.job_event(), {"kind": "complete", "metrics": {}}]
        with patch("zkinfer.runtime.preparation_runner.subprocess.Popen",
                   side_effect=self.fake_child(events)):
            with self.assertLogs(self.builder.logger, level="INFO") as logs:
                self.run_child()
        self.assertIn("Calibrating circuit", "\n".join(logs.output))

    def test_crash_after_partial_publication_is_failure(self):
        events = [{"kind": "plan", "count": 2}, self.job_event()]
        published = []
        with patch("zkinfer.runtime.preparation_runner.subprocess.Popen",
                   side_effect=self.fake_child(events, code=9)):
            with self.assertRaisesRegex(RuntimeError, "exited with code 9"):
                self.run_child(on_job_ready=published.append)
        self.assertEqual(len(published), 1)

    def test_incomplete_success_is_rejected(self):
        events = [{"kind": "plan", "count": 2}, self.job_event(),
                  {"kind": "complete", "metrics": {}}]
        with patch("zkinfer.runtime.preparation_runner.subprocess.Popen",
                   side_effect=self.fake_child(events)):
            with self.assertRaisesRegex(RuntimeError, "incomplete plan"):
                self.run_child()

"""Fresh-process request preparation with incremental, file-backed IPC.

Only small JSON descriptors cross the process boundary. The parent owns
scheduler state, leases and SQLite. Only explicit progress messages share the event channel.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import resource
import tempfile
import traceback

from zkinfer.config.runtime import SetupStoreConfig
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.runtime.state_store import CoordinatorStateStore
from zkinfer.storage.shared_store import SharedStorage


def emit(directory, index, event):
    destination = directory / f"{index:08d}.json"
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".event-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(event, file)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(task_file, event_dir):
    task = json.loads(Path(task_file).read_text())
    events = Path(event_dir)
    request = CoordinatorStateStore._request_from_dict(task["request"])
    index = 0

    def send(kind, **payload):
        nonlocal index
        emit(events, index, {"kind": kind, **payload})
        index += 1

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            send("progress", message=record.getMessage())

    progress = logging.getLogger("zkinfer.preparation.progress")
    handler = ProgressHandler()
    progress.addHandler(handler)
    try:
        builder = RequestBuilder(logging.getLogger("preparation"), **task["builder"])
        shared = SharedStorage(**task["storage"])
        streaming = task["streaming"]
        jobs = builder.build_jobs(
            request=request, shared_storage=shared,
            setup_store=SetupStoreConfig(**task["setup_store"]),
            max_retries=task["max_retries"],
            on_plan=lambda count: send("plan", count=count),
            on_job_ready=(lambda job: send("job", job=CoordinatorStateStore._job_to_dict(job)))
            if streaming else None,
        )
        if not streaming:
            for job in jobs:
                send("job", job=CoordinatorStateStore._job_to_dict(job))
        send("complete", metrics={
            **request.preparation_metrics,
            "preparation_process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        return 0
    except Exception as exc:
        traceback.print_exc()
        send("failed", error=str(exc) or type(exc).__name__,
             metrics=request.preparation_metrics)
        return 1
    finally:
        progress.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--event-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(run(args.task_file, args.event_dir))

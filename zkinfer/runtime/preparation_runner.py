"""Parent-side subprocess supervision; no expensive model operations here."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import tempfile

from zkinfer.runtime.rayon_config import rayon_environment
from zkinfer.runtime.state_store import CoordinatorStateStore


def run_preparation(request, builder, shared, setup_store, max_retries,
                    rayon_threads, *, on_plan=None, on_job_ready=None):
    run_dir = Path(request.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "request_preparation.log"
    jobs = []
    expected = None
    seen = set()
    terminal = None
    started = time.perf_counter()
    logger = builder.logger
    logger.info("Request %s | preparing model", request.request_id)
    with tempfile.TemporaryDirectory(prefix=".preparation-", dir=run_dir) as tmp:
        directory = Path(tmp)
        events = directory / "events"
        events.mkdir()
        task = directory / "task.json"
        task.write_text(json.dumps({
            "request": CoordinatorStateStore._request_to_dict(request),
            "builder": {name: getattr(builder, name) for name in (
                "witness_preparation", "max_propagated_scale", "allow_boundary_rescale",
                "preparation_cache_enabled", "prover_threads")},
            "storage": {"backend": shared.backend, "setup_root": shared.setup_root,
                        "requests_root": shared.requests_root, "s3_bucket": shared.s3_bucket},
            "setup_store": asdict(setup_store),
            "max_retries": max_retries,
            "streaming": on_job_ready is not None,
        }))
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "zkinfer.runtime.preparation_process",
                 "--task-file", str(task), "--event-dir", str(events)],
                stdout=log, stderr=subprocess.STDOUT,
                env=rayon_environment(rayon_threads),
            )
            index = 0
            try:
                while True:
                    path = events / f"{index:08d}.json"
                    if not path.exists():
                        if proc.poll() is not None:
                            # Recheck after observing exit: the final atomic rename
                            # may have happened between exists() and poll().
                            if path.exists():
                                continue
                            break
                        time.sleep(0.05)
                        continue
                    event = json.loads(path.read_text())
                    path.unlink()
                    index += 1
                    if terminal is not None:
                        raise RuntimeError("Unexpected event after preparation terminal event")
                    kind = event["kind"]
                    if kind == "plan":
                        count = event["count"]
                        if expected is not None or type(count) is not int or count < 1:
                            raise RuntimeError("Invalid preparation plan")
                        expected = count
                        logger.info("Request %s | planned %d proof jobs", request.request_id, count)
                        if on_plan is not None:
                            on_plan(count)
                    elif kind == "progress":
                        logger.info("Request %s | %s", request.request_id, event["message"])
                    elif kind == "job":
                        job = CoordinatorStateStore._job_from_dict(event["job"])
                        if (expected is None or len(jobs) >= expected
                                or job.inference_request_id != request.request_id
                                or job.job_id in seen):
                            raise RuntimeError("Invalid preparation job event")
                        seen.add(job.job_id)
                        jobs.append(job)
                        if on_job_ready is not None:
                            on_job_ready(job)
                        logger.info("Request %s | job ready %d/%d | %s", request.request_id, len(jobs), expected, job.job_id)
                    elif kind in {"complete", "failed"}:
                        terminal = event
                    else:
                        raise RuntimeError(f"Unknown preparation event: {kind}")
                code = proc.wait()
                if terminal:
                    # Child has a snapshot of the initial metrics. Do not replace
                    # parent-owned dispatch/queue measurements with stale values.
                    request.preparation_metrics.update({
                        k: v for k, v in terminal.get("metrics", {}).items()
                        if k not in {"first_job_ready_time(s)", "preparation_queue_wait_time(s)"}
                    })
                request.preparation_metrics["preparation_subprocess_wall_time(s)"] = (
                    time.perf_counter() - started)
                if code != 0 or terminal is None or terminal["kind"] != "complete":
                    detail = terminal.get("error", "") if terminal else ""
                    raise RuntimeError(f"Preparation subprocess exited with code {code}: {detail}; see {log_path}")
                if expected is None or len(jobs) != expected:
                    raise RuntimeError("Preparation subprocess returned an incomplete plan")
                logger.info("Request %s | preparation complete | %d jobs | %.2fs", request.request_id, len(jobs), time.perf_counter() - started)
                return jobs
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

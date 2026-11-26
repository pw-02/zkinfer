"""Run request verification outside the coordinator interpreter."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from zkinfer.runtime.rayon_config import rayon_environment
from zkinfer.runtime.state_store import CoordinatorStateStore


def run_verification(request, verifier, rayon_threads):
    root = Path(request.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "verification_process.log"
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".verification-", dir=root) as tmp:
        task = Path(tmp) / "task.json"
        result = Path(tmp) / "result.json"
        task.write_text(json.dumps({
            "request": CoordinatorStateStore._request_to_dict(request),
            "jobs": [CoordinatorStateStore._job_to_dict(job) for job in request.proof_jobs],
            "verifier": {
                "srs_dir": str(verifier.srs_dir),
                "enabled": verifier.enabled,
                "require_split_chain": verifier.require_split_chain,
                "write_request_manifest": False,
                "trusted_vk_manifest": str(verifier.trusted_vk_manifest) if verifier.trusted_vk_manifest else None,
            },
        }))
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "zkinfer.verification.verification_process",
                 str(task), str(result)],
                stdout=log, stderr=subprocess.STDOUT,
                env=rayon_environment(rayon_threads),
            )
            try:
                code = proc.wait()
                if code != 0 or not result.is_file():
                    raise RuntimeError(f"Verification subprocess exited with code {code}; see {log_path}")
                report = json.loads(result.read_text())
                if (report.get("request_id") != request.request_id
                        or type(report.get("valid")) is not bool
                        or report.get("enabled") is not True):
                    raise RuntimeError("Invalid verification subprocess response")
                if report["valid"] and len(report.get("job_reports", [])) != len(request.proof_jobs):
                    raise RuntimeError("Verification subprocess returned an incomplete report")
                report["verification_subprocess_wall_time_s"] = time.perf_counter() - started
                return report
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

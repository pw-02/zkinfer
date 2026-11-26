"""Fresh interpreter for native proof verification and boundary checks."""
from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path

from zkinfer.runtime.state_store import CoordinatorStateStore
from zkinfer.verification.verifier import RuntimeVerifier


def main():
    task_path, result_path = map(Path, sys.argv[1:])
    task = json.loads(task_path.read_text())
    request = CoordinatorStateStore._request_from_dict(task["request"])
    request.proof_jobs = [CoordinatorStateStore._job_from_dict(job) for job in task["jobs"]]
    report = RuntimeVerifier(**task["verifier"]).verify_request(request).to_dict()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["verifier_resources"] = {
        "pid": os.getpid(),
        "peak_rss_kib": usage.ru_maxrss,
        "user_cpu_time_s": usage.ru_utime,
        "system_cpu_time_s": usage.ru_stime,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }
    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report))
    os.replace(tmp, result_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskRunner


class _ExpectedCalibrationCandidateFilter(logging.Filter):
    """Suppress expected EZKL calibration-search candidate rejections."""

    _EXPECTED_PREFIXES = (
        "calibration failed extended k is too large",
        "calibration failed max lookup input",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "ezkl.execute" or record.levelno < logging.ERROR:
            return True
        message = record.getMessage()
        return not message.startswith(self._EXPECTED_PREFIXES)


def _write_json_atomic(path: str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=destination.name + ".",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, destination)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _persist_task_metadata(task: LocalProverTask) -> None:
    """Keep the effective per-job task after the temporary workdir is removed."""
    _write_json_atomic(
        str(Path(task.output_dir) / "task_config.json"),
        task.to_dict(),
    )


def _normalize_disabled_admission_metrics(
    task: LocalProverTask,
    metrics: dict,
) -> None:
    """Do not report synthetic stage predictions when admission is disabled."""
    if task.stage_admission_enabled:
        return

    keep = {
        "stage_admission_enabled",
        "stage_admission_wait_time(s)",
    }
    for key in list(metrics):
        if key.startswith("stage_admission_") and key not in keep:
            metrics.pop(key, None)
    metrics["stage_admission_enabled"] = False
    metrics["stage_admission_wait_time(s)"] = 0.0


def run_task(task_file: str, result_file: str) -> int:
    started = time.perf_counter()
    with open(task_file, "r", encoding="utf-8") as file:
        task = LocalProverTask.from_dict(json.load(file))

    logger = logging.getLogger(f"prover-process-{task.job_id}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    calibration_filter = _ExpectedCalibrationCandidateFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(calibration_filter)

    _persist_task_metadata(task)

    try:
        result = ProverTaskRunner(logger=logger).run(task)
        _normalize_disabled_admission_metrics(task, result.metrics)
        payload = {
            "ok": True,
            "pid": os.getpid(),
            "wall_time_s": time.perf_counter() - started,
            "result": result.to_dict(),
        }
        _write_json_atomic(result_file, payload)
        return 0
    except Exception as exc:
        logger.exception("Prover process failed")
        (Path(task.output_dir) / "status.txt").write_text("FAILED", encoding="utf-8")
        _write_json_atomic(
            result_file,
            {
                "ok": False,
                "pid": os.getpid(),
                "wall_time_s": time.perf_counter() - started,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one isolated zkInfer prover task"
    )
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--result_file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_task(args.task_file, args.result_file))


if __name__ == "__main__":
    main()

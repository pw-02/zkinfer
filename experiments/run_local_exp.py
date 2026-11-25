#!/usr/bin/env python3
"""Run a local zkInfer experiment suite.

One NodeSupervisor represents this machine and may run multiple isolated prover
subprocesses. Experiment-facing names are deliberately descriptive; this module
translates them into the runtime RPC fields where necessary.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


class ActiveRunLog:
    def __init__(self) -> None:
        self.file = None
        self.lock = threading.Lock()

    def set_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.close()
            self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> None:
        with self.lock:
            if self.file is not None:
                self.file.write(text)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class ActiveRunLogHandler(logging.Handler):
    def __init__(self, active_log: ActiveRunLog) -> None:
        super().__init__()
        self.active_log = active_log

    def emit(self, record: logging.LogRecord) -> None:
        self.active_log.write(self.format(record) + "\n")


def setup_logger(active_log: ActiveRunLog) -> logging.Logger:
    logger = logging.getLogger("exp")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = ActiveRunLogHandler(active_log)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    env: dict
    logger: logging.Logger
    active_log: ActiveRunLog
    proc: Optional[subprocess.Popen] = None
    tee_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.logger.info("Starting %s: %s", self.name, " ".join(self.command))
        self.proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self.env,
            text=True,
            bufsize=1,
        )
        self.tee_thread = threading.Thread(
            target=self._tee_output,
            name=f"{self.name}-tee",
            daemon=True,
        )
        self.tee_thread.start()

    def _tee_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            prefixed = f"[{self.name}] {line}"
            print(prefixed, end="")
            self.active_log.write(prefixed)

    def returncode(self) -> Optional[int]:
        return None if self.proc is None else self.proc.poll()

    def stop(self, timeout_sec: int = 30) -> bool:
        if self.proc is None:
            return True
        graceful = True
        if self.proc.poll() is None:
            self.logger.info("Stopping %s", self.name)
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                graceful = False
                self.proc.kill()
                self.proc.wait(timeout=timeout_sec)
        if self.tee_thread is not None:
            self.tee_thread.join(timeout=timeout_sec)
        graceful = graceful and self.proc.returncode == 0
        if graceful:
            self.logger.info("%s exited successfully", self.name)
        else:
            self.logger.warning("%s did not shut down cleanly (exit code %s)", self.name, self.proc.returncode)
        return graceful


class ProcessGroup:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processes: List[ManagedProcess] = []

    def add(self, process: ManagedProcess) -> None:
        self.processes.append(process)
        process.start()

    def stop_all(self) -> bool:
        graceful = True
        for process in reversed(self.processes):
            graceful = process.stop() and graceful
        return graceful

    def raise_if_any_failed(self) -> None:
        failed = [
            process
            for process in self.processes
            if process.returncode() is not None and process.returncode() != 0
        ]
        if failed:
            details = "\n".join(
                f"- {process.name} exited with code {process.returncode()}"
                for process in failed
            )
            raise RuntimeError(f"One or more runtime processes failed:\n{details}")


def make_env(cfg: DictConfig) -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f".:{existing}" if existing else "."
    env["PYTHONUNBUFFERED"] = "1"
    env["ZKINFER_KEEP_PROOF_ARTIFACTS"] = (
        "true" if bool(cfg.output.keep_proof_artifacts) else "false"
    )
    env["ZKINFER_PROFILING_MODE"] = str(cfg.profiling.mode)
    return env


def common_overrides(cfg: DictConfig) -> List[str]:
    requests_dir = Path(str(cfg.paths.runs_dir)) / "requests"
    logs_dir = Path(str(cfg.paths.runs_dir)) / "logs"
    return [
        "resources.coordinator_threads=" + str(cfg.get("resources", {}).get("coordinator_threads") or "null"),
        "resources.prover_threads=" + str(cfg.get("resources", {}).get("prover_threads") or "null"),
        f"coordinator.host={cfg.launch.coordinator_host}",
        f"coordinator.port={cfg.launch.coordinator_port}",
        f"coordinator.runs_dir={requests_dir}",
        f"coordinator.logs_dir={logs_dir}",
        f"coordinator.state_db_path={Path(str(cfg.paths.runs_dir)) / 'coordinator_state.sqlite3'}",
        f"coordinator.scheduling_policy={cfg.coordinator.scheduling_policy}",
        f"coordinator.witness_preparation={cfg.coordinator.get('witness_preparation', 'upfront')}",
        f"coordinator.preparation_dispatch={cfg.coordinator.get('preparation_dispatch', 'batch')}",
        f"coordinator.preparation_workers={cfg.coordinator.get('preparation_workers', 1)}",
        f"coordinator.preparation_max_pending={cfg.coordinator.get('preparation_max_pending', 8)}",
        "coordinator.preparation_cache_enabled=" + (
            "null" if cfg.coordinator.get("preparation_cache_enabled") is None
            else str(cfg.coordinator.preparation_cache_enabled).lower()
        ),
        f"coordinator.max_propagated_scale={cfg.coordinator.get('max_propagated_scale', 16)}",
        f"coordinator.allow_boundary_rescale={str(bool(cfg.coordinator.get('allow_boundary_rescale', True))).lower()}",
        f"verification.enabled={str(bool(cfg.get('verification', {}).get('enabled', True))).lower()}",
        "verification.trusted_vk_manifest=" + str(
            cfg.get("verification", {}).get("trusted_vk_manifest") or "null"
        ),
        f"node.runs_dir={requests_dir}",
        f"node.logs_dir={logs_dir}",
        f"node.runtime_state_dir={cfg.node.get('runtime_state_dir', 'storage/node/runtime')}",
        f"storage.request_data.path={cfg.storage.request_data.path}",
        f"storage.reusable_setup.path={cfg.storage.reusable_setup.path}",
        f"storage.reusable_setup.enabled={str(bool(cfg.storage.reusable_setup.enabled)).lower()}",
        f"storage.resource_profiles.enabled={str(bool(cfg.storage.get('resource_profiles', {}).get('enabled', True))).lower()}",
        f"storage.resource_profiles.update_history={str(bool(cfg.storage.get('resource_profiles', {}).get('update_history', True))).lower()}",
        "storage.resource_profiles.path=" + (
            str(cfg.storage.get("resource_profiles", {}).get("path"))
            if cfg.storage.get("resource_profiles", {}).get("path") is not None else "null"
        ),
    ]


def coordinator_command(cfg: DictConfig) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.coordinator_grpc",
        *common_overrides(cfg),
    ]


def node_command(cfg: DictConfig) -> List[str]:
    admission = cfg.node.stage_admission
    overrides = [
        f"node.node_id={cfg.node.node_id}",
        f"node.resource_sampling_interval_sec={cfg.node.resource_sampling_interval_sec}",
        f"node.stage_admission.enabled={str(bool(admission.enabled)).lower()}",
        f"node.stage_admission.mode={admission.get('mode', 'stage')}",
        f"node.stage_admission.memory_safety_reserve_gb={admission.memory_safety_reserve_gb}",
        f"node.stage_admission.history_percentile={admission.history_percentile}",
    ]
    if cfg.node.get("max_concurrent_jobs", None) is not None:
        overrides.append(f"node.max_concurrent_jobs={int(cfg.node.max_concurrent_jobs)}")
    else:
        overrides.append("node.max_concurrent_jobs=null")

    return [
        sys.executable,
        "-u",
        "-m",
        "zkinfer.runtime.node.supervisor",
        *common_overrides(cfg),
        *overrides,
    ]


def wait_for_port(
    host: str,
    port: int,
    timeout_sec: int,
    process_group: ProcessGroup,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        process_group.raise_if_any_failed()
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    process_group.raise_if_any_failed()
    raise TimeoutError(f"Coordinator did not become ready at {host}:{port}")


def wait_for_port_closed(host: str, port: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                time.sleep(0.25)
        except OSError:
            return


def wait_for_request(
    client: ZKInferenceClient,
    request_id: str,
    poll_interval_sec: int,
    process_group: ProcessGroup,
    logger: logging.Logger,
) -> None:
    previous = None
    while True:
        process_group.raise_if_any_failed()
        status = client.get_request_status(request_id)
        snapshot = (
            status.status,
            status.total_jobs,
            status.completed_jobs,
            status.in_progress_jobs,
            status.queued_jobs,
            status.failed_jobs,
        )
        if snapshot != previous:
            logger.info(
                "Request %s | status=%s | completed=%s/%s | running=%s | queued=%s | failed=%s",
                request_id,
                status.status,
                status.completed_jobs,
                status.total_jobs,
                status.in_progress_jobs,
                status.queued_jobs,
                status.failed_jobs,
            )
            previous = snapshot
        if status.status == "completed":
            return
        if status.status in {"failed", "unknown"}:
            raise RuntimeError(f"Request {request_id} {status.status}: {status.message}")
        time.sleep(poll_interval_sec)


def load_workload_cfg(workload_name: str) -> DictConfig:
    path = Path(__file__).parent / "config" / "workload" / f"{workload_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Workload config not found: {path}")
    return OmegaConf.load(path)


def _runtime_partition_mode(cfg: DictConfig) -> str:
    return "fixed" if bool(cfg.execution.partitioning_enabled) else "none"


def submit_workload(
    client: ZKInferenceClient,
    cfg: DictConfig,
    workload_cfg: DictConfig,
) -> str:
    partition_size = int(cfg.execution.partition_size)
    if partition_size < 1:
        raise ValueError("execution.partition_size must be at least one")
    return client.submit_inference_request(
        name=workload_cfg.name,
        onnx_model_path=workload_cfg.onnx_file,
        input_data_path=workload_cfg.input_file,
        split_mode=_runtime_partition_mode(cfg),
        ops_per_chunk=partition_size,
        simplify_model=cfg.execution.get("simplify_model", False),
        input_shapes=workload_cfg.get("input_shapes", None),
    )


def save_request_metadata(
    cfg: DictConfig,
    workload_cfg: DictConfig,
    request_id: str,
) -> Path:
    request_dir = Path(str(cfg.paths.runs_dir)) / "requests" / request_id
    config_dir = request_dir / "config"
    logs_dir = request_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=config_dir / "experiment_config.yaml")
    OmegaConf.save(config=workload_cfg, f=config_dir / "workload_config.yaml")
    (config_dir / "runtime_partitioning.txt").write_text(
        f"enabled={str(bool(cfg.execution.partitioning_enabled)).lower()}\n"
        f"partition_size={int(cfg.execution.partition_size)}\n"
        f"runtime_mode={_runtime_partition_mode(cfg)}\n",
        encoding="utf-8",
    )
    return logs_dir


def cleanup_request_store(
    cfg: DictConfig,
    logger: logging.Logger,
    request_ids: List[str],
) -> None:
    if bool(cfg.output.keep_request_data):
        logger.info("Keeping canonical request inputs and proof outputs after experiment run")
        return

    request_root = Path(str(cfg.storage.request_data.path)).resolve()
    protected = [
        Path(str(cfg.storage.reusable_setup.path)).resolve(),
        Path(str(cfg.paths.runs_dir)).resolve(),
    ]
    for request_id in set(request_ids):
        if not request_id or Path(request_id).name != request_id or request_id in {".", ".."}:
            raise ValueError("Invalid request ID for cleanup")
        request_dir = request_root / request_id
        if any(request_dir == path or request_dir in path.parents for path in protected):
            raise ValueError("Request cleanup would remove reusable setup or experiment records")
        if request_dir.exists():
            shutil.rmtree(request_dir)
        logger.info(
            "Deleted canonical request data for %s; reusable setup and experiment records retained",
            request_id,
        )


def prune_empty_run_dirs(run_dir: Path, request_ids: List[str]) -> None:
    """Remove only empty, run-owned temporary directories."""
    root = run_dir.resolve()
    targets = [root / "storage"]
    targets.extend(root / "requests" / rid / "tmp" for rid in request_ids
                   if rid and Path(rid).name == rid and rid not in (".", ".."))
    for target in targets:
        if not target.is_dir() or target.is_symlink():
            continue
        for current, dirs, files in os.walk(target, topdown=False, followlinks=False):
            path = Path(current)
            if path.is_symlink():
                continue
            try:
                path.rmdir()
            except OSError:
                pass  # Never remove nonempty directories.


def run(cfg: DictConfig) -> None:
    active_log = ActiveRunLog()
    logger = setup_logger(active_log)
    group = ProcessGroup(logger)
    env = make_env(cfg)
    submitted_request_ids: List[str] = []
    successful = False

    try:
        group.add(
            ManagedProcess(
                name="coordinator",
                command=coordinator_command(cfg),
                env=env,
                logger=logger,
                active_log=active_log,
            )
        )
        wait_for_port(
            cfg.launch.coordinator_host,
            int(cfg.launch.coordinator_port),
            int(cfg.launch.coordinator_ready_timeout_sec),
            group,
        )

        group.add(
            ManagedProcess(
                name=str(cfg.node.node_id),
                command=node_command(cfg),
                env=env,
                logger=logger,
                active_log=active_log,
            )
        )
        time.sleep(float(cfg.launch.node_startup_delay_sec))

        client = ZKInferenceClient(
            target=f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"
        )
        failures = []

        for workload_name in cfg.suite.workloads:
            try:
                workload_cfg = load_workload_cfg(workload_name)
                logger.info("Running workload: %s", workload_cfg.name)
                request_id = submit_workload(client, cfg, workload_cfg)
                submitted_request_ids.append(request_id)
                logs_dir = save_request_metadata(cfg, workload_cfg, request_id)
                active_log.set_path(logs_dir / "run.log")
                logger.info("Submitted request %s", request_id)
                wait_for_request(
                    client,
                    request_id,
                    int(cfg.launch.poll_completion_interval_sec),
                    group,
                    logger,
                )
                logger.info("Finished workload: %s", workload_cfg.name)
            except Exception as exc:
                logger.exception("Workload failed: %s", workload_name)
                failures.append((workload_name, str(exc)))
            finally:
                active_log.close()

        if failures:
            for name, error in failures:
                logger.warning("FAILED %s: %s", name, error)
            raise RuntimeError("One or more workloads failed")
        successful = True
        logger.info("All workloads completed")
    finally:
        active_log.close()
        stopped_cleanly = group.stop_all()
        wait_for_port_closed(
            cfg.launch.coordinator_host,
            int(cfg.launch.coordinator_port),
            10,
        )
        reports_ok = True
        try:
            from zkinfer.profiling.collect_reports import collect
            if (Path(str(cfg.paths.runs_dir)) / "requests").is_dir():
                logger.info("Generating request profiling reports (including partial results)")
                for report, count in collect(Path(str(cfg.paths.runs_dir))):
                    logger.info("Saved %s (%d phase occurrences)", report, count)
        except Exception:
            reports_ok = False
            logger.exception("Report collection failed; retaining diagnostic storage")
        if successful and stopped_cleanly and reports_ok:
            logger.info("Cleaning temporary storage according to output settings")
            cleanup_request_store(cfg, logger, submitted_request_ids)
            root = Path(str(cfg.paths.runs_dir)).resolve() / "storage"
            if not cfg.output.get("keep_cache", True) and (root / "cache").exists():
                shutil.rmtree(root / "cache")
            if (root / "runtime").exists():
                shutil.rmtree(root / "runtime")
            prune_empty_run_dirs(Path(str(cfg.paths.runs_dir)), submitted_request_ids)
            logger.info("Experiment completed successfully")
            logger.info("Results saved to %s", cfg.paths.runs_dir)
        else:
            logger.warning("Run failed, was interrupted, or required forced shutdown; retaining diagnostic storage")
            if successful:
                raise RuntimeError("Workloads completed but shutdown or reporting failed; cleanup skipped")


@hydra.main(config_path="./config", config_name="runtime_experiment", version_base=None)
def advanced_main(cfg: DictConfig) -> None:
    run(cfg)


def main():
    # Preserve existing Hydra overrides for advanced scripts.
    if any("=" in arg for arg in sys.argv[1:]):
        advanced_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config/config.yaml")
    parser.add_argument("--workload")
    args = parser.parse_args()
    from experiments.scripts.launcher_config import (
        configure_run_paths,
        load_config,
        make_run_name,
    )

    user, cfg = load_config(args.config, workload=args.workload)
    run_dir = Path(str(cfg.paths.runs_dir)) / make_run_name(user)
    configure_run_paths(cfg, user, run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config").mkdir()
    OmegaConf.save(user, run_dir / "config" / "experiment.yaml")
    OmegaConf.save(cfg, run_dir / "config" / "effective.yaml", resolve=True)
    print(f"Run directory: {run_dir}", flush=True)
    run(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations
from zkinfer.profiling.run_metadata import save_process_metadata, copy_run_metadata

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import grpc
import fcntl
import hashlib
import hydra
import psutil
from omegaconf import DictConfig

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.config.runtime import RuntimeConfig, build_runtime_config
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.artifact_transfer import ArtifactTransfer
from zkinfer.runtime.node.stage_admission import (
    StageAdmissionController,
    StageResourceHistory,
)
from zkinfer.runtime.node.prover_task import LocalProverTask, ProverTaskResult
from zkinfer.utils.network import get_ip
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.profiling.execution_trace import write_execution_trace
from zkinfer.profiling.job_features import save_circuit_features


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def from_pb_artifact(ref) -> ArtifactRef:
    if ref is None or not getattr(ref, "backend", "") or not getattr(ref, "uri", ""):
        return ArtifactRef.empty()
    return ArtifactRef(
        backend=ref.backend,
        uri=ref.uri,
        sha256=ref.sha256,
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket,
    )


def to_pb_artifact(ref: Optional[ArtifactRef]):
    if ref is None or ref.is_empty:
        return pb.ArtifactRef()
    return pb.ArtifactRef(
        backend=ref.backend or "",
        uri=ref.uri or "",
        sha256=ref.sha256 or "",
        size_bytes=int(ref.size_bytes or 0),
        s3_bucket=ref.s3_bucket or "",
    )


class NodeSupervisor:
    """One control agent per machine.

    The coordinator assigns whole proof jobs. Each proof executes in an isolated
    subprocess. ``max_concurrent_jobs`` is an optional explicit whole-job cap.
    When it is ``None``, the node uses live resource state and stage admission;
    provisional lease reservations prevent the polling loop from leasing a large
    burst of jobs before their process memory becomes visible to the OS.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        logger: Optional[logging.Logger] = None,
        resolved_config=None,
    ):
        self.cfg = runtime_config
        self.metadata_snapshot = save_process_metadata(
            Path(self.cfg.node.runs_dir).parent / "config",
            "node",
            runtime_config,
            resolved_config,
        )
        self.node_id = self.cfg.node.node_id or get_ip()
        self.logger = logger or logging.getLogger("node-supervisor")
        self.target = f"{self.cfg.coordinator.host}:{self.cfg.coordinator.port}"
        self.poll_interval_sec = self.cfg.node.job_poll_interval_sec
        self.stage_cfg = self.cfg.node.stage_admission
        self.grpc_max_message_bytes = self.cfg.coordinator.grpc_max_message_mb * 1024 * 1024

        self.stage_admission_dir = Path(self.cfg.node.runtime_state_dir)
        self.stage_admission_dir.mkdir(parents=True, exist_ok=True)
        self.resource_profiles = StageResourceHistory(
            self.stage_admission_dir / "resource_profiles"
        )
        self.admission_probe = StageAdmissionController(
            root_dir=str(self.stage_admission_dir),
            setup_key="__supervisor__",
            enabled=self.stage_cfg.enabled,
            fallback_memory_gb=0.0,
            memory_safety_reserve_gb=self.stage_cfg.memory_safety_reserve_gb,
            history_percentile=self.stage_cfg.history_percentile,
            use_history=self.cfg.setup_store.resource_profiles_enabled,
        )

        self.active_jobs: Dict[str, threading.Thread] = {}
        self.active_tasks: Dict[str, LocalProverTask] = {}
        self.lease_memory_reservations_gb: Dict[str, float] = {}
        self.active_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.channel = None
        self.stub = None

    def connect(self) -> None:
        if self.channel:
            self.channel.close()
        self.channel = grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", self.grpc_max_message_bytes),
                ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
            ],
        )
        self.stub = pb_grpc.ZKJobServiceStub(self.channel)
        self.logger.info("Node %s connected to coordinator at %s", self.node_id, self.target)

    def reconnect(self) -> None:
        self.logger.info("Reconnecting to coordinator...")
        self.connect()
        time.sleep(self.poll_interval_sec)

    def safe_grpc_call(self, call_fn, action: str, max_retries: int = 3):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return call_fn()
            except grpc.RpcError as exc:
                last_error = exc
                self.logger.error(
                    "%s failed, attempt %d/%d: %s code=%s",
                    action,
                    attempt,
                    max_retries,
                    exc,
                    exc.code(),
                )
                self.reconnect()
            except Exception as exc:
                last_error = exc
                self.logger.error(
                    "%s failed, attempt %d/%d: %s",
                    action,
                    attempt,
                    max_retries,
                    exc,
                    exc_info=True,
                )
                time.sleep(self.poll_interval_sec)
        raise last_error

    def active_job_ids(self) -> list[str]:
        with self.active_lock:
            return sorted(self.active_jobs.keys())

    def active_job_count(self) -> int:
        with self.active_lock:
            return len(self.active_jobs)

    @staticmethod
    def _read_task_stage(task: LocalProverTask) -> str:
        status_file = Path(task.output_dir) / "status.txt"
        try:
            return status_file.read_text(encoding="utf-8").strip() or "STARTING"
        except OSError:
            return "STARTING"

    def active_stages(self) -> Dict[str, str]:
        with self.active_lock:
            items = list(self.active_tasks.items())
        return {job_id: self._read_task_stage(task) for job_id, task in items}

    def _has_unprofiled_active_job(self) -> bool:
        with self.active_lock:
            return any(
                estimate <= 0.0
                for job_id, estimate in self.lease_memory_reservations_gb.items()
                if job_id in self.active_jobs
            )

    def _provisional_start_memory_gb(self) -> float:
        with self.active_lock:
            reservations = dict(self.lease_memory_reservations_gb)
            tasks = dict(self.active_tasks)

        total = 0.0
        for job_id, estimate in reservations.items():
            if estimate <= 0.0:
                continue
            task = tasks.get(job_id)
            if task is None or self._read_task_stage(task) == "STARTING":
                total += estimate
        return total

    def can_accept_more_work(self) -> bool:
        max_concurrent_jobs = self.cfg.node.max_concurrent_jobs
        if max_concurrent_jobs is not None:
            if self.active_job_count() >= max_concurrent_jobs:
                return False
        elif self._has_unprofiled_active_job():
            return False

        memory = psutil.virtual_memory()
        available_gb = float(memory.available) / (1024**3)
        provisional_gb = self._provisional_start_memory_gb()
        return (
            available_gb - provisional_gb
            > self.stage_cfg.memory_safety_reserve_gb
        )

    def build_node_status(self):
        memory = psutil.virtual_memory()
        cpu_total = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        cpu_percent = psutil.cpu_percent(interval=None)
        actual_available_gb = float(memory.available) / (1024**3)
        actual_cpu_available = max(0.0, float(cpu_total) * (1.0 - cpu_percent / 100.0))

        reservations = (
            self.admission_probe.current_reservations()
            if self.stage_cfg.enabled
            else {"memory_gb": 0.0, "count": 0.0}
        )
        provisional_start_gb = self._provisional_start_memory_gb()
        total_memory_gb = float(memory.total) / (1024**3)
        resource_available_gb = max(
            0.0,
            total_memory_gb
            - self.stage_cfg.memory_safety_reserve_gb
            - float(reservations.get("memory_gb") or 0.0)
            - provisional_start_gb,
        )
        reported_available_gb = min(
            max(
                0.0,
                actual_available_gb
                - self.stage_cfg.memory_safety_reserve_gb
                - provisional_start_gb,
            ),
            resource_available_gb,
        )

        return pb.NodeStatus(
            node_id=self.node_id,
            memory_total_gb=total_memory_gb,
            memory_available_gb=reported_available_gb,
            cpu_total_cores=float(cpu_total),
            cpu_available_cores=actual_cpu_available,
            cache_limit_gb=0.0,
            cache_used_gb=0.0,
            active_job_ids=self.active_job_ids(),
            cached_setup_keys=[],
        )

    def poll_for_job(self):
        return self.safe_grpc_call(
            lambda: self.stub.PollNode(self.build_node_status()),
            action="PollNode",
        )

    def submit_job_result(
        self,
        lease,
        status: str,
        artifact_manifest_ref: Optional[ArtifactRef] = None,
        message: str = "",
        metrics: Optional[Dict] = None,
    ) -> bool:
        def _call():
            return self.stub.SubmitJobResult(
                pb.NodeJobResult(
                    node_id=self.node_id,
                    lease_id=lease.lease_id,
                    request_id=lease.request_id,
                    job_id=lease.job_id,
                    status=status,
                    message=message or "",
                    perf_metrics_json=json.dumps(metrics or {}),
                    artifact_manifest=to_pb_artifact(artifact_manifest_ref),
                )
            )

        ack = self.safe_grpc_call(_call, action=f"SubmitJobResult({status})")
        if not ack.ok:
            self.logger.warning(
                "Coordinator rejected %s for job %s lease=%s",
                status,
                lease.job_id,
                lease.lease_id,
            )
            return False
        self.logger.info("Submitted %s for job %s", status, lease.job_id)
        return True

    def send_heartbeat(self) -> None:
        stages = self.active_stages()

        def _call():
            return self.stub.SendNodeHeartbeat(
                pb.NodeHeartbeat(
                    node_id=self.node_id,
                    active_job_ids=self.active_job_ids(),
                    status="RUNNING",
                    message=json.dumps({"stages": stages}, sort_keys=True),
                )
            )

        self.safe_grpc_call(_call, action="SendNodeHeartbeat", max_retries=1)

    def heartbeat_loop(self) -> None:
        while not self.stop_event.wait(max(1, self.cfg.heartbeat.interval_sec)):
            try:
                self.send_heartbeat()
            except Exception:
                self.logger.warning("Node heartbeat failed", exc_info=True)

    def start_lease(self, lease) -> None:
        with self.active_lock:
            if lease.job_id in self.active_jobs:
                self.logger.warning("Ignoring duplicate lease for active job %s", lease.job_id)
                return
            self.lease_memory_reservations_gb[lease.job_id] = max(
                0.0,
                float(lease.predicted_peak_memory_gb or 0.0),
            )
            thread = threading.Thread(
                target=self._run_lease_thread,
                args=(lease,),
                name=f"prover-controller-{lease.job_id[:8]}",
                daemon=False,
            )
            self.active_jobs[lease.job_id] = thread
            thread.start()
        self.logger.info(
            "Started prover subprocess controller for job %s lease=%s predicted_peak=%.2fGB",
            lease.job_id,
            lease.lease_id,
            float(lease.predicted_peak_memory_gb or 0.0),
        )

    def _run_lease_thread(self, lease) -> None:
        try:
            self.run_lease(lease)
        finally:
            with self.active_lock:
                self.active_jobs.pop(lease.job_id, None)
                self.active_tasks.pop(lease.job_id, None)
                self.lease_memory_reservations_gb.pop(lease.job_id, None)

    def _run_prover_subprocess(self, task: LocalProverTask) -> ProverTaskResult:
        task_file = Path(task.tmp_dir) / "prover_task.json"
        result_file = Path(task.tmp_dir) / "prover_result.json"
        log_file = Path(task.output_dir) / "prover_process.log"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task_file, "w", encoding="utf-8") as file:
            json.dump(task.to_dict(), file, indent=2)

        started = time.perf_counter()
        with open(log_file, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "zkinfer.runtime.node.prover_process",
                    "--task_file",
                    str(task_file),
                    "--result_file",
                    str(result_file),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            return_code = proc.wait()
        wall_time = time.perf_counter() - started

        if not result_file.exists():
            raise RuntimeError(
                f"Prover subprocess exited with code {return_code} without a result; see {log_file}"
            )
        with open(result_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not payload.get("ok"):
            raise RuntimeError(
                f"Prover subprocess failed ({payload.get('error_type', 'error')}): "
                f"{payload.get('error', 'unknown error')}; see {log_file}"
            )
        result = ProverTaskResult.from_dict(payload["result"])
        result.metrics["prover_subprocess_wall_time(s)"] = wall_time
        result.metrics["prover_subprocess_pid"] = int(payload.get("pid") or 0)
        return result

    @staticmethod
    def _validate_job_component(value: str) -> None:
        if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
            raise ValueError("Request, job and lease IDs must be safe path components")

    def _job_paths(self, lease):
        for value in (lease.request_id, lease.job_id, lease.lease_id):
            self._validate_job_component(value)
        request_dir = Path(self.cfg.node.runs_dir) / lease.request_id
        return (
            request_dir,
            request_dir / "tmp" / lease.job_id,
            request_dir / "artifacts" / "jobs" / lease.job_id,
        )

    @contextmanager
    def _job_guard(self, lease):
        request_dir, _, _ = self._job_paths(lease)
        key = hashlib.sha256(
            str((request_dir / lease.job_id).resolve()).encode()
        ).hexdigest()
        lock_dir = self.stage_admission_dir / "job_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _cleanup_published_attempt(self, task):
        from zkinfer.storage.s3 import delete_file

        for name in (
            "proof.pf",
            "witness.json",
            "vk.json",
            "settings.json",
            "manifest.json",
        ):
            filename = f"{task.lease_id}--{name}"
            if task.output_backend == "filesystem":
                (Path(task.output_prefix) / filename).unlink(missing_ok=True)
            elif task.output_backend == "s3":
                delete_file(
                    task.output_s3_bucket,
                    f"{task.output_prefix.rstrip('/')}/{filename}",
                )

    def run_lease(self, lease) -> None:
        with self._job_guard(lease):
            self._run_lease(lease)

    def _run_lease(self, lease) -> None:
        local_task = None
        _, tmp_dir, output_dir = self._job_paths(lease)
        failed = False
        completion_attempted = False
        cleanup_published = False
        try:
            local_task = self.prepare_local_task(lease)
            with self.active_lock:
                self.active_tasks[lease.job_id] = local_task

            result = self._run_prover_subprocess(local_task)
            setup_publish_metrics = {}
            if self.cfg.setup_store.enabled and not local_task.setup_read_only:
                transfer = ArtifactTransfer(tmp_dir / "transfer")
                setup_publish_metrics = transfer.publish_setup(
                    from_pb_artifact(lease.setup_cache), local_task.setup_key,
                    local_task.setup_dir, bool(local_task.prepared_witness_path),
                )

            update_history = self.cfg.setup_store.resource_profiles_update_history
            if update_history is None:
                update_history = self.cfg.setup_store.resource_profiles_enabled
            if update_history:
                self.resource_profiles.update_from_metrics(local_task.setup_key, result.metrics)

            metrics = {
                "node_id": self.node_id,
                "node_cache_bytes": 0,
                **result.metrics,
                **self._flatten_setup_publish_metrics(setup_publish_metrics),
            }
            completion_attempted = True
            accepted = self.submit_job_result(
                lease=lease,
                status="COMPLETED",
                artifact_manifest_ref=result.artifact_manifest_ref,
                message=result.message,
                metrics=metrics,
            )
            if accepted is False:
                failed = True
                cleanup_published = True
        except Exception as exc:
            failed = not completion_attempted
            cleanup_published = not completion_attempted
            self.logger.error(
                "Job %s failed on node %s: %s",
                lease.job_id,
                self.node_id,
                exc,
                exc_info=True,
            )
            partial = {"node_id": self.node_id}
            if local_task is not None:
                for operation in (
                    lambda: save_circuit_features(local_task),
                    lambda: write_execution_trace(output_dir),
                ):
                    try:
                        operation()
                    except Exception:
                        self.logger.warning("Partial profiling checkpoint failed", exc_info=True)
                try:
                    partial.update(parse_resource_usage_file(str(output_dir / "resource_usage.csv")))
                except Exception:
                    self.logger.warning("Partial RSS metrics unavailable", exc_info=True)
            self.submit_job_result(
                lease=lease,
                status="FAILED",
                message=str(exc),
                metrics=partial,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if failed and output_dir.is_dir():
                # Keep attempt diagnostics outside canonical job directories;
                # a retry may recreate the latter.
                archive = output_dir.parent.parent / "failed_attempts" / lease.job_id / lease.lease_id
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    output_dir.rename(archive)
            if cleanup_published and local_task:
                try:
                    self._cleanup_published_attempt(local_task)
                except Exception:
                    self.logger.warning(
                        "Could not remove failed job's published files",
                        exc_info=True,
                    )

    def prepare_local_task(self, lease, _legacy_context=None) -> LocalProverTask:
        prepare_started = time.perf_counter()
        lease_id = lease.lease_id
        request_dir, tmp_dir, output_dir = self._job_paths(lease)
        for directory in (tmp_dir, output_dir):
            if directory.is_symlink():
                directory.unlink()
            elif directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        copy_run_metadata(self.metadata_snapshot, request_dir)

        setup_key = lease.setup_key or lease.job_id
        transfer = ArtifactTransfer(tmp_dir / "transfer")
        model_path, model_metrics = transfer.materialize(from_pb_artifact(lease.model), "model.onnx")
        input_path, input_metrics = transfer.materialize(from_pb_artifact(lease.input), "input.json")
        model_materialize_time_s = float(model_metrics["materialize_time(s)"])
        input_materialize_time_s = float(input_metrics["materialize_time(s)"])
        setup_dir = str(tmp_dir / "setup")
        Path(setup_dir).mkdir(parents=True, exist_ok=True)
        prepared = from_pb_artifact(lease.input).uri.endswith(".prepared.json")
        setup_metrics = {"setup_key": setup_key, "bytes_downloaded": 0,
                         "bytes_from_local_cache": 0, "restore_time(s)": 0.0}
        if self.cfg.setup_store.enabled:
            setup_metrics = transfer.restore_setup(
                from_pb_artifact(lease.setup_cache), setup_key, setup_dir, prepared
            )

        prepared_witness_path = None
        prepared_metrics = {}
        if from_pb_artifact(lease.input).uri.endswith(".prepared.json"):
            from zkinfer.backends.ezkl.preparation import PREPARED_INPUT_KIND, file_digest, read_json
            descriptor_ref = from_pb_artifact(lease.input)
            if not descriptor_ref.sha256 or file_digest(input_path) != descriptor_ref.sha256:
                raise ValueError("Prepared descriptor hash mismatch")
            descriptor = read_json(input_path)
            if (
                descriptor.get("kind") != PREPARED_INPUT_KIND
                or descriptor.get("request_id") != lease.request_id
                or descriptor.get("setup_key") != setup_key
            ):
                raise ValueError("Prepared descriptor does not match job")
            refs = {}
            for name in ("settings", "compiled", "witness"):
                ref = ArtifactRef.from_dict(descriptor[name])
                if ref.is_empty or not ref.sha256:
                    raise ValueError(f"Unbound prepared artifact: {name}")
                local, metrics = transfer.materialize(ref, name)
                if file_digest(local) != ref.sha256:
                    raise ValueError(f"Prepared artifact hash mismatch: {name}")
                refs[name] = (Path(local), ref)
                prepared_metrics[name] = metrics

            # The setup identity binds the exact settings and compiled bytes.
            # Never overwrite conflicting cached setup/key material.
            for name, filename in (("settings", "settings.json"), ("compiled", "network.compiled")):
                source, ref = refs[name]
                target = Path(setup_dir) / filename
                if target.exists():
                    if file_digest(target) != ref.sha256:
                        raise ValueError(f"Prepared setup cache mismatch: {filename}")
                else:
                    if setup_metrics.get("shared_read_only"):
                        raise ValueError(f"Missing read-only prepared setup: {filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + "." + lease_id + ".tmp")
                    try:
                        shutil.copyfile(source, temporary)
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
            prepared_witness_path = str(refs["witness"][0])

        output_ref = from_pb_artifact(lease.output_prefix)
        transfer_metrics = {
            "model": model_metrics,
            "input": input_metrics,
            "setup": setup_metrics,
            "prepared": prepared_metrics,
            "prepare_local_task_time(s)": time.perf_counter() - prepare_started,
        }

        return LocalProverTask(
            node_id=self.node_id,
            request_id=lease.request_id,
            job_id=lease.job_id,
            lease_id=lease_id,
            setup_key=setup_key,
            model_path=model_path,
            input_path=input_path,
            setup_dir=setup_dir,
            setup_read_only=bool(setup_metrics.get("shared_read_only", False)),
            prepared_witness_path=prepared_witness_path,
            tmp_dir=str(tmp_dir),
            output_dir=str(output_dir),
            output_prefix=output_ref.uri or str(output_dir),
            output_backend=output_ref.backend or self.cfg.request_store.backend,
            output_s3_bucket=output_ref.s3_bucket
            or self.cfg.request_store.s3_bucket
            or "",
            resource_sampling_interval_sec=self.cfg.node.resource_sampling_interval_sec,
            model_materialize_time_s=model_materialize_time_s,
            input_materialize_time_s=input_materialize_time_s,
            transfer_metrics=transfer_metrics,
            stage_admission_dir=str(self.stage_admission_dir),
            stage_admission_enabled=self.stage_cfg.enabled,
            admission_mode=self.stage_cfg.mode,
            memory_safety_reserve_gb=self.stage_cfg.memory_safety_reserve_gb,
            history_percentile=self.stage_cfg.history_percentile,
            predicted_peak_memory_gb=float(lease.predicted_peak_memory_gb or 0.0),
            use_resource_history=self.cfg.setup_store.resource_profiles_enabled,
        )

    @staticmethod
    def _flatten_setup_publish_metrics(metrics: Dict) -> Dict[str, float | int]:
        return {
            "transfer_setup_bytes_uploaded": int(metrics.get("bytes_uploaded", 0) or 0),
            "transfer_setup_publish_time(s)": float(
                metrics.get("publish_time(s)", 0.0) or 0.0
            ),
            "transfer_pk_compress_time(s)": float(
                metrics.get("pk_compress_time(s)", 0.0) or 0.0
            ),
            "transfer_setup_publish_eviction_count": int(
                metrics.get("eviction_count", 0) or 0
            ),
            "transfer_setup_publish_evicted_bytes": int(
                metrics.get("evicted_bytes", 0) or 0
            ),
        }

    def _join_active_threads(self) -> None:
        while True:
            with self.active_lock:
                threads = list(self.active_jobs.values())
            if not threads:
                return
            for thread in threads:
                thread.join(timeout=1)

    def run(self) -> None:
        self.connect()
        heartbeat_thread = threading.Thread(
            target=self.heartbeat_loop,
            name="node-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            while True:
                if not self.can_accept_more_work():
                    time.sleep(self.poll_interval_sec)
                    continue
                lease = self.poll_for_job()
                if not lease.job_available:
                    time.sleep(self.poll_interval_sec)
                    continue
                self.start_lease(lease)
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested; draining active proof jobs")
        finally:
            # Keep heartbeats alive while outstanding leases finish.
            try:
                self._join_active_threads()
            finally:
                self.stop_event.set()
                heartbeat_thread.join(timeout=5)
                if self.channel:
                    self.channel.close()
            self.logger.info("Node supervisor stopped successfully")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    runtime_config = build_runtime_config(cfg=cfg, process_role="node")
    node_id = cfg.node.node_id or get_ip()
    logger = setup_logger(
        "node-supervisor",
        log_file=f"{runtime_config.node.logs_dir}/node_{node_id}.log",
    )
    logger.info("Starting NodeSupervisor...")
    NodeSupervisor(
        runtime_config=runtime_config,
        logger=logger,
        resolved_config=cfg,
    ).run()


if __name__ == "__main__":
    main()


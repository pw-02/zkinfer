import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import grpc
import hydra
import psutil
from omegaconf import DictConfig

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc
from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
from zkinfer.config.runtime import RuntimeConfig, build_runtime_config
from zkinfer.profiling.msm_fft_parser import (
    read_csv_first_row,
    summarize_fft_report,
    summarize_msm_report,
)
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.storage.s3 import download_file
from zkinfer.utils.network import get_ip

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

@dataclass
class LocalJobPaths:
    model_path: str
    input_path: str
    tmp_dir: str
    artifact_dir: str
    cache_prefix: str
    model_s3_read_time: float = 0.0
    input_s3_read_time: float = 0.0


class ZKProofWorker:
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = runtime_config
        self.logger = logger or logging.getLogger("worker")

        self.worker_id = self.cfg.worker.worker_id or get_ip()
        self.target = f"{self.cfg.coordinator.host}:{self.cfg.coordinator.port}"

        self.poll_interval_sec = self.cfg.worker.job_poll_interval_sec
        self.resource_monitor_interval_sec = (
            self.cfg.worker.resource_monitor_interval_sec
        )
        self.send_proofs_to_coordinator = self.cfg.worker.send_proofs_to_coordinator

        self.grpc_max_message_bytes = (
            self.cfg.coordinator.grpc_max_message_mb * 1024 * 1024
        )

        self.channel = None
        self.stub = None

        self.num_cpus_physical = psutil.cpu_count(logical=False)
        self.num_cpus_logical = psutil.cpu_count(logical=True)
        self.mem_gb = round(psutil.virtual_memory().total / 1e9, 2)

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
        self.logger.info("Connected to coordinator at %s", self.target)

    def reconnect(self) -> None:
        self.logger.info("Attempting to reconnect to coordinator...")
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

    def get_next_job(self):
        return self.safe_grpc_call(
            lambda: self.stub.GetNextJob(
                pb.WorkerRequest(worker_id=self.worker_id)
            ),
            action="GetNextJob",
        )

    def submit_job_result(
        self,
        job_id: str,
        status: str,
        proof: Optional[bytes] = None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _call():
            return self.stub.SubmitJobResult(
                pb.JobResult(
                    worker_id=self.worker_id,
                    job_id=job_id,
                    proof=proof or b"",
                    status=status,
                    message=message or "",
                    perf_metrics_json=json.dumps(perf_metrics) if perf_metrics else "",
                )
            )

        self.safe_grpc_call(
            _call,
            action=f"SubmitJobResult({status})",
        )

        self.logger.info("Submitted final result for job %s: %s", job_id, status)

    def prepare_local_paths(self, assignment) -> LocalJobPaths:
        job_id = assignment.job_id

        model_path = assignment.model_path
        input_path = assignment.input_path

        worker_request_dir = Path(self.cfg.worker.runs_dir) / assignment.request_id

        tmp_dir = worker_request_dir / "tmp" / job_id
        artifact_dir = worker_request_dir / "artifacts" / "jobs" / job_id

        tmp_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cache_prefix = assignment.cache_path or self.cfg.proving_cache.root_dir

        if self.cfg.file_transfer.backend != "s3":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path does not exist: {input_path}")

            return LocalJobPaths(
                model_path=model_path,
                input_path=input_path,
                tmp_dir=str(tmp_dir),
                artifact_dir=str(artifact_dir),
                cache_prefix=cache_prefix,
            )

        if not self.cfg.file_transfer.s3_bucket:
            raise ValueError("storage.s3_bucket must be set when storage.backend=s3")

        local_model_path = tmp_dir / os.path.basename(model_path)
        local_input_path = tmp_dir / "input.json"

        start = time.perf_counter()
        download_file(
            self.cfg.file_transfer.s3_bucket,
            model_path,
            str(local_model_path),
        )
        model_s3_read_time = time.perf_counter() - start

        start = time.perf_counter()
        download_file(
            self.cfg.file_transfer.s3_bucket,
            input_path,
            str(local_input_path),
        )
        input_s3_read_time = time.perf_counter() - start

        return LocalJobPaths(
            model_path=str(local_model_path),
            input_path=str(local_input_path),
            tmp_dir=str(tmp_dir),
            artifact_dir=str(artifact_dir),
            cache_prefix=cache_prefix,
            model_s3_read_time=model_s3_read_time,
            input_s3_read_time=input_s3_read_time,
        )

    def start_monitoring_processes(
        self,
        job_id: str,
        artifact_dir: str,
        status_file: str,
    ):
        worker_pid = str(os.getpid())
        resource_usage_file = os.path.join(artifact_dir, "resource_usage.csv")

        resource_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.profiling.resource_logger",
                "--output_file",
                resource_usage_file,
                "--pid",
                worker_pid,
                "--interval",
                str(self.resource_monitor_interval_sec),
            ]
        )

        heartbeat_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.runtime.heartbeat",
                "--target",
                self.target,
                "--worker_id",
                self.worker_id,
                "--job_id",
                job_id,
                "--status_file",
                status_file,
                "--parent_pid",
                worker_pid,
                "--interval_sec",
                str(self.cfg.heartbeat.interval_sec),
            ]
        )

        return heartbeat_proc, resource_proc, resource_usage_file

    def stop_processes(self, *processes) -> None:
        for proc in processes:
            if not proc:
                continue

            proc.terminate()

            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    def collect_metrics(
        self,
        artifact_dir: str,
        resource_usage_file: str,
        proof_stages: EZKLProofStages,
        ezkl_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            resource_metrics = parse_resource_usage_file(resource_usage_file)

        except Exception as exc:
            self.logger.error(
                "Failed to parse resource usage file: %s",
                exc,
                exc_info=True,
            )

            resource_metrics = {
                "max_process_memory(GB)": None,
                "max_system_memory(GB)": None,
                "avg_process_cpu_raw(%)": None,
                "avg_process_cpu_machine(%)": None,
                "avg_system_cpu(%)": None,
            }

        return {
            "worker_id": self.worker_id,
            **resource_metrics,
            "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
            "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
            **read_csv_first_row(os.path.join(artifact_dir, "halo2_circuit.csv")),
            **read_csv_first_row(os.path.join(artifact_dir, "halo2_prover_cpu.csv")),
            **summarize_fft_report(os.path.join(artifact_dir, "halo2_ffts.csv")),
            **summarize_msm_report(os.path.join(artifact_dir, "halo2_msms.csv")),
            **ezkl_metrics,
        }

    def read_proof_bytes(self, proof_path: str) -> bytes:
        if not self.send_proofs_to_coordinator:
            return b""

        if not os.path.exists(proof_path):
            raise FileNotFoundError(f"Proof file does not exist: {proof_path}")

        with open(proof_path, "rb") as file:
            return file.read()

    def build_proof_stages(
        self,
        local_paths: LocalJobPaths,
        status_file: str,
    ) -> EZKLProofStages:
        return EZKLProofStages(
            input_data_path=local_paths.input_path,
            onnx_model_path=local_paths.model_path,
            artifact_dir=local_paths.artifact_dir,
            proving_cache_enabled=self.cfg.proving_cache.enabled,
            proving_cache_overwrite=self.cfg.proving_cache.overwrite,
            proving_cache_type=self.cfg.proving_cache.backend,
            proving_cache_root_dir=local_paths.cache_prefix,
            proving_cache_s3_bucket=self.cfg.proving_cache.s3_bucket,
            proving_cache_s3_prefix=local_paths.cache_prefix,
            status_file=status_file,
            local_tmp_dir=local_paths.tmp_dir,
            logger=self.logger,
        )

    def run_assignment(self, assignment) -> None:
        job_id = assignment.job_id

        local_paths = None
        heartbeat_proc = None
        resource_proc = None

        try:
            local_paths = self.prepare_local_paths(assignment)
            status_file = os.path.join(local_paths.artifact_dir, "status.txt")

            (
                heartbeat_proc,
                resource_proc,
                resource_usage_file,
            ) = self.start_monitoring_processes(
                job_id=job_id,
                artifact_dir=local_paths.artifact_dir,
                status_file=status_file,
            )

            proof_stages = self.build_proof_stages(
                local_paths=local_paths,
                status_file=status_file,
            )

            ezkl_metrics, ezkl_settings = proof_stages.run_all(setup_only=False)

            #disabling proof upload for now to save time and space during development - can re-enable later if needed

            # if os.path.exists(proof_stages.proof_path):
            #     shutil.copy(
            #         proof_stages.proof_path,
            #         os.path.join(local_paths.artifact_dir, "proof.pf"),
            #     )

            # Preserve the request-specific proof and witness before tmp cleanup.
            proof_output_path = os.path.join(
                local_paths.artifact_dir,
                "proof.pf",
            )

            witness_output_path = os.path.join(
                local_paths.artifact_dir,
                "witness.json",
            )

            if not os.path.exists(proof_stages.proof_path):
                raise FileNotFoundError(
                    f"Proof file was not generated: {proof_stages.proof_path}"
                )

            if not os.path.exists(proof_stages.witness_path):
                raise FileNotFoundError(
                    f"Witness file was not generated: {proof_stages.witness_path}"
                )

            shutil.copy2(
                proof_stages.proof_path,
                proof_output_path,
            )

            shutil.copy2(
                proof_stages.witness_path,
                witness_output_path,
            )

            self.logger.info(
                "Preserved proof at %s",
                proof_output_path,
            )

            self.logger.info(
                "Preserved witness at %s",
                witness_output_path,
            )

            

            settings_output_path = os.path.join(
                local_paths.artifact_dir,
                "ezkl_settings.json",
            )

            with open(settings_output_path, "w", encoding="utf-8") as file:
                json.dump(ezkl_settings, file, indent=4)

            metrics = self.collect_metrics(
                artifact_dir=local_paths.artifact_dir,
                resource_usage_file=resource_usage_file,
                proof_stages=proof_stages,
                ezkl_metrics={
                    **ezkl_metrics,
                    "model_s3_read_time(s)": local_paths.model_s3_read_time,
                    "input_s3_read_time(s)": local_paths.input_s3_read_time,
                },
            )

            self.submit_job_result(
                job_id=job_id,
                status="COMPLETED",
                proof=self.read_proof_bytes(proof_stages.proof_path),
                perf_metrics=metrics,
                message="Proof computation completed successfully",
            )

            self.logger.info("Completed job %s", job_id)

        except Exception as exc:
            self.logger.error(
                "Job %s failed: %s",
                job_id,
                exc,
                exc_info=True,
            )

            self.submit_job_result(
                job_id=job_id,
                status="FAILED",
                message=str(exc),
            )

        finally:
            self.stop_processes(
                heartbeat_proc,
                resource_proc,
            )

            if local_paths and os.path.exists(local_paths.tmp_dir):
                self.logger.info(
                    "Cleaning up temporary directory: %s",
                    local_paths.tmp_dir,
                )

                shutil.rmtree(
                    local_paths.tmp_dir,
                    ignore_errors=True,
                )

    def run_once(self) -> None:
        assignment = self.get_next_job()

        if not assignment.job_available:
            self.logger.info("No jobs available. Sleeping...")
            time.sleep(self.poll_interval_sec)
            return

        self.logger.info("Received job %s", assignment.job_id)
        self.run_assignment(assignment)

    def run(self) -> None:
        self.connect()

        while True:
            self.run_once()


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    runtime_config = build_runtime_config(cfg=cfg, process_role="worker")
    worker_id = cfg.worker.worker_id or get_ip()
    logger = setup_logger("worker",log_file=f"{runtime_config.worker.logs_dir}/worker_{worker_id}.log")
    logger.info("Starting ZKProofWorker...")

    worker = ZKProofWorker(
        runtime_config=runtime_config,
        logger=logger,
    )

    worker.run()


if __name__ == "__main__":
    main()

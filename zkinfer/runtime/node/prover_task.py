from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
from zkinfer.profiling.msm_fft_parser import (
    read_csv_first_row,
    summarize_fft_report,
    summarize_msm_report,
)
from zkinfer.profiling.resource_parser import parse_resource_usage_file
from zkinfer.profiling.halo2_phases import write_phase_report
from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.stage_admission import StageAdmissionController
from zkinfer.storage.s3 import upload_file


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalProverTask:
    node_id: str
    request_id: str
    job_id: str
    setup_key: str
    model_path: str
    input_path: str
    setup_dir: str
    tmp_dir: str
    output_dir: str
    output_prefix: str
    output_backend: str
    output_s3_bucket: str = ""
    setup_read_only: bool = False
    resource_sampling_interval_sec: float = 1.0
    model_materialize_time_s: float = 0.0
    input_materialize_time_s: float = 0.0
    transfer_metrics: Dict[str, Any] = field(default_factory=dict)
    stage_admission_dir: str = ""
    stage_admission_enabled: bool = True
    admission_mode: str = "stage"
    memory_safety_reserve_gb: float = 8.0
    history_percentile: float = 0.90
    predicted_peak_memory_gb: float = 0.0
    lease_id: str = ""
    keep_proof_artifacts: bool = field(
        default_factory=lambda: _env_bool("ZKINFER_KEEP_PROOF_ARTIFACTS", False)
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalProverTask":
        return cls(**dict(data))


@dataclass(frozen=True)
class ProverTaskResult:
    artifact_manifest_ref: Optional[ArtifactRef]
    metrics: Dict[str, Any]
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        ref = self.artifact_manifest_ref
        return {
            "artifact_manifest_ref": ref.to_dict() if ref else None,
            "metrics": self.metrics,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProverTaskResult":
        ref_data = data.get("artifact_manifest_ref")
        ref = ArtifactRef.from_dict(ref_data) if ref_data else None
        if ref is not None and ref.is_empty:
            ref = None
        return cls(
            artifact_manifest_ref=ref,
            metrics=data.get("metrics") or {},
            message=data.get("message") or "",
        )


class StageAwareEZKLProofStages(EZKLProofStages):
    """EZKL stages guarded by node-local memory admission."""

    def __init__(
        self,
        *args,
        admission: StageAdmissionController,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.admission = admission
        self.admission_metrics: Dict[str, Any] = {
            "stage_admission_enabled": admission.enabled,
            "admission_mode": admission.mode,
            "stage_admission_wait_time(s)": 0.0,
        }
        self._transition_started = time.perf_counter()
        self._transition_path = (
            Path(self.status_file).with_name("stage_transitions.csv")
            if self.status_file
            else None
        )

    @staticmethod
    def _metric_stage(stage: str) -> str:
        return stage.lower().replace("-", "_")

    def _update_status(self, stage: str) -> None:
        super()._update_status(stage)
        self._record_transition(stage)

    def _record_transition(self, stage, timestamp=None, elapsed_s=None):
        if self._transition_path is None:
            return
        self._transition_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._transition_path.exists()
        with self._transition_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["timestamp", "elapsed_s", "stage", "event"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp or datetime.now(timezone.utc).isoformat(
                        timespec="microseconds"
                    ),
                    "elapsed_s": (
                        time.perf_counter() - self._transition_started
                        if elapsed_s is None
                        else elapsed_s
                    ),
                    "stage": stage,
                    "event": "enter",
                }
            )
            file.flush()

    @contextmanager
    def _admit(self, stage: str) -> Iterator[Dict[str, Any]]:
        self._update_status(f"WAITING_FOR_{stage}")
        with self.admission.hold(stage) as info:
            metric_stage = self._metric_stage(stage)
            wait_time = float(info.get("wait_time_s") or 0.0)
            self.admission_metrics["stage_admission_wait_time(s)"] += wait_time
            self.admission_metrics[
                f"stage_admission_{metric_stage}_wait_time(s)"
            ] = self.admission_metrics.get(
                f"stage_admission_{metric_stage}_wait_time(s)",
                0.0,
            ) + wait_time
            self.admission_metrics[
                f"stage_admission_{metric_stage}_predicted_memory(GB)"
            ] = float(info.get("memory_gb") or 0.0)
            self.admission_metrics[
                f"stage_admission_{metric_stage}_estimate_samples"
            ] = int(info.get("samples") or 0)
            self.admission_metrics[
                f"stage_admission_{metric_stage}_estimate_source"
            ] = str(info.get("source") or "fallback")
            yield info

    def calibrate_settings_with_fallback(self):
        with self._admit("CALIBRATING"):
            return super().calibrate_settings_with_fallback()

    def compile_circuit(self):
        with self._admit("COMPILING"):
            return super().compile_circuit()

    def get_srs(self):
        with self._admit("GETTING_SRS"):
            return super().get_srs()

    def gen_witness(self):
        with self._admit("GENERATING_WITNESS"):
            return super().gen_witness()

    def gen_keys(self):
        with self._admit("KEY_GEN"):
            return super().gen_keys()

    def compute_proof(self):
        with self._admit("PROVING"):
            return super().compute_proof()

    def setup_and_compute_proof(self) -> Dict[str, float]:
        with self._admit("FUSED_KEYGEN_PROVE"):
            self._update_status("FUSED_KEYGEN_PROVE")
            try:
                return super().setup_and_compute_proof()
            finally:
                trace = (
                    Path(self.status_file).with_name("fused_stage_transitions.csv")
                    if self.status_file
                    else None
                )
                available = trace is not None and trace.exists()
                self.admission_metrics["ezkl_fused_stage_trace_available"] = available
                if available:
                    with trace.open(newline="", encoding="utf-8") as file:
                        for row in csv.DictReader(file):
                            timestamp = datetime.fromtimestamp(
                                int(row["timestamp_unix_us"]) / 1_000_000,
                                timezone.utc,
                            ).isoformat(timespec="microseconds")
                            self._record_transition(
                                row["stage"],
                                timestamp,
                                float(row["elapsed_s"]),
                            )
                else:
                    self.logger.warning(
                        "EZKL wheel lacks detailed fused stage events; "
                        "combined execution is recorded. Rebuild the wheel."
                    )


class ProverTaskRunner:
    """Execute one proof using local paths only."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def run(self, task: LocalProverTask) -> ProverTaskResult:
        os.makedirs(task.output_dir, exist_ok=True)
        os.makedirs(task.tmp_dir, exist_ok=True)

        status_file = os.path.join(task.output_dir, "status.txt")
        resource_proc, resource_usage_file = self._start_resource_logger(
            task,
            status_file,
        )

        try:
            proof_stages = self._build_proof_stages(task, status_file)
            with open(
                os.path.join(task.output_dir, "admission_profile_before.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    proof_stages.admission.profiles.load(task.setup_key),
                    file,
                    indent=2,
                    sort_keys=True,
                )

            # Enter before preparation and retain the reservation through output
            # publication. The outer finally also releases it after exceptions.
            if task.stage_admission_enabled and task.admission_mode == "whole_job":
                proof_stages._update_status("WAITING_FOR_WHOLE_JOB")
            with proof_stages.admission.hold_job() as job_info:
                return self._run_admitted(task, resource_usage_file, proof_stages, job_info)
        finally:
            self._stop_process(resource_proc)

    def _run_admitted(
        self, task: LocalProverTask, resource_usage_file: str,
        proof_stages: StageAwareEZKLProofStages, job_info: Dict[str, Any],
    ) -> ProverTaskResult:
        ezkl_metrics, _ = proof_stages.run_all(setup_only=False)
        ezkl_metrics.update(proof_stages.admission_metrics)
        ezkl_metrics["whole_job_admission_wait_time(s)"] = float(
            job_info.get("wait_time_s", 0.0)
        )
        ezkl_metrics["whole_job_admission_predicted_memory(GB)"] = float(
            job_info.get("memory_gb", 0.0)
        )

        metrics = self._collect_metrics(
            task=task,
            resource_usage_file=resource_usage_file,
            proof_stages=proof_stages,
            ezkl_metrics=ezkl_metrics,
        )
        manifest_ref, publish_metrics = self._publish_job_artifacts(
            task,
            proof_stages,
        )
        metrics.update(self._flatten_transfer_metrics(task.transfer_metrics))
        metrics.update(self._flatten_output_metrics(publish_metrics))
        metrics["prover_process_pid"] = os.getpid()
        return ProverTaskResult(
            artifact_manifest_ref=manifest_ref,
            metrics=metrics,
            message="Proof computation completed successfully",
        )

    def _start_resource_logger(self, task: LocalProverTask, status_file: str):
        resource_usage_file = os.path.join(task.output_dir, "resource_usage.csv")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zkinfer.profiling.resource_logger",
                "--output_file",
                resource_usage_file,
                "--pid",
                str(os.getpid()),
                "--interval",
                str(task.resource_sampling_interval_sec),
                "--status_file",
                status_file,
                "--stage_admission_dir",
                task.stage_admission_dir,
                "--include_children",
                "--exclude_pid",
                str(os.getpid()),
            ]
        )
        return process, resource_usage_file

    @staticmethod
    def _stop_process(process) -> None:
        if not process:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except Exception:
            process.kill()

    def _build_proof_stages(
        self,
        task: LocalProverTask,
        status_file: str,
    ) -> StageAwareEZKLProofStages:
        admission_dir = task.stage_admission_dir or os.path.join(
            task.tmp_dir,
            "stage-admission",
        )
        admission = StageAdmissionController(
            root_dir=admission_dir,
            setup_key=task.setup_key,
            enabled=task.stage_admission_enabled,
            mode=task.admission_mode,
            fallback_memory_gb=task.predicted_peak_memory_gb,
            memory_safety_reserve_gb=task.memory_safety_reserve_gb,
            history_percentile=task.history_percentile,
        )
        return StageAwareEZKLProofStages(
            input_data_path=task.input_path,
            onnx_model_path=task.model_path,
            artifact_dir=task.output_dir,
            proving_cache_enabled=True,
            proving_cache_overwrite=False,
            proving_cache_type="filesystem",
            proving_cache_root_dir=task.setup_dir,
            proving_cache_read_only=task.setup_read_only,
            proving_cache_s3_bucket=None,
            proving_cache_s3_prefix=None,
            status_file=status_file,
            local_tmp_dir=task.tmp_dir,
            logger=self.logger,
            admission=admission,
        )

    @staticmethod
    def _prediction_error_metrics(
        resource_metrics: Dict[str, Any],
        admission_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        def observed(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and number >= 0 else None

        def stage_actual(stage, metric):
            stages = (
                ("key_gen", "proving", "fused_keygen_prove")
                if stage == "fused_keygen_prove"
                else (stage,)
            )
            values = [
                observed(resource_metrics.get(f"stage_{name}_{metric}"))
                for name in stages
            ]
            values = [value for value in values if value is not None]
            return max(values) if values else None

        result: Dict[str, Any] = {}
        prefix = "stage_admission_"
        suffix = "_predicted_memory(GB)"
        for key, predicted_raw in admission_metrics.items():
            if not (key.startswith(prefix) and key.endswith(suffix)):
                continue
            stage = key[len(prefix) : -len(suffix)]
            predicted = observed(predicted_raw)
            actual = stage_actual(stage, "max_memory(GB)")
            result[f"stage_admission_{stage}_actual_memory(GB)"] = actual
            if actual is not None and predicted is not None:
                error = predicted - actual
                result[f"stage_admission_{stage}_memory_error(GB)"] = error
                if actual > 0:
                    result[
                        f"stage_admission_{stage}_memory_abs_pct_error"
                    ] = abs(error) / actual * 100.0
        return result

    def _collect_metrics(
        self,
        task: LocalProverTask,
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

        try:
            phase_metrics = write_phase_report(task.output_dir)
        except Exception as exc:
            self.logger.warning("Failed to correlate Halo2 phases: %s", exc)
            phase_metrics = {"halo2_phase_trace_error": str(exc)}

        prediction_metrics = self._prediction_error_metrics(
            resource_metrics,
            ezkl_metrics,
        )
        return {
            "node_id": task.node_id,
            "setup_key": task.setup_key,
            "predicted_peak_memory(GB)": task.predicted_peak_memory_gb,
            "resource_sampling_interval(s)": task.resource_sampling_interval_sec,
            "model_materialize_time(s)": task.model_materialize_time_s,
            "input_materialize_time(s)": task.input_materialize_time_s,
            **resource_metrics,
            **phase_metrics,
            "pk_file_size(GB)": proof_stages.get_pk_file_size_gb(),
            "vk_file_size(GB)": proof_stages.get_vk_file_size_gb(),
            **read_csv_first_row(os.path.join(task.output_dir, "halo2_circuit.csv")),
            **read_csv_first_row(os.path.join(task.output_dir, "halo2_prover_cpu.csv")),
            **summarize_fft_report(os.path.join(task.output_dir, "halo2_ffts.csv")),
            **summarize_msm_report(os.path.join(task.output_dir, "halo2_msms.csv")),
            **ezkl_metrics,
            **prediction_metrics,
        }

    @staticmethod
    def _same_path(a: str | Path, b: str | Path) -> bool:
        try:
            return Path(a).resolve() == Path(b).resolve()
        except OSError:
            return os.path.abspath(str(a)) == os.path.abspath(str(b))

    def _publish_file(
        self,
        local_path: str,
        task: LocalProverTask,
        artifact_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        started = time.perf_counter()
        size_bytes = os.path.getsize(local_path)
        published_name = (
            f"{task.lease_id}--{artifact_name}" if task.lease_id else artifact_name
        )
        if task.output_backend == "filesystem":
            output_path = os.path.join(task.output_prefix, published_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not self._same_path(local_path, output_path):
                shutil.copy2(local_path, output_path)
            return output_path, {
                "artifact": artifact_name,
                "backend": "filesystem",
                "bytes_uploaded": 0,
                "bytes_written": size_bytes,
                "publish_time(s)": time.perf_counter() - started,
            }

        if task.output_backend == "s3":
            if not task.output_s3_bucket:
                raise ValueError("S3 output artifact requires a bucket")
            key = f"{task.output_prefix.rstrip('/')}/{published_name}"
            upload_file(local_path, task.output_s3_bucket, key)
            return key, {
                "artifact": artifact_name,
                "backend": "s3",
                "bytes_uploaded": size_bytes,
                "bytes_written": 0,
                "publish_time(s)": time.perf_counter() - started,
            }

        raise ValueError(f"Unsupported output backend: {task.output_backend}")

    def _publish_job_artifacts(
        self,
        task: LocalProverTask,
        proof_stages: EZKLProofStages,
    ) -> Tuple[ArtifactRef, Dict[str, Any]]:
        artifact_sources = {
            "proof.pf": proof_stages.proof_path,
            "witness.json": proof_stages.witness_path,
            "vk.json": proof_stages.vk_path,
            "settings.json": proof_stages.settings_path,
        }
        for artifact_name, source_path in artifact_sources.items():
            if not source_path or not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Required artifact {artifact_name} was not generated: {source_path}"
                )

        artifact_metadata: Dict[str, Dict[str, Any]] = {}
        publish_metrics: Dict[str, Any] = {
            "artifacts": {},
            "output_bytes_uploaded": 0,
            "output_bytes_written": 0,
            "output_publish_time(s)": 0.0,
        }
        tmp_dir = Path(task.tmp_dir).resolve()

        for artifact_name, source_path in artifact_sources.items():
            source = Path(source_path).resolve()
            publish_source = source
            retained_path: Optional[str] = None

            if task.keep_proof_artifacts:
                output_path = os.path.join(task.output_dir, artifact_name)
                try:
                    source.relative_to(tmp_dir)
                    source_is_temporary = True
                except ValueError:
                    source_is_temporary = False

                if not self._same_path(source, output_path):
                    if source_is_temporary:
                        shutil.move(str(source), output_path)
                    else:
                        shutil.copy2(str(source), output_path)
                publish_source = Path(output_path)
                retained_path = output_path

            size_bytes = publish_source.stat().st_size
            published_uri, file_metrics = self._publish_file(
                str(publish_source),
                task,
                artifact_name,
            )
            publish_metrics["artifacts"][artifact_name] = file_metrics
            publish_metrics["output_bytes_uploaded"] += int(
                file_metrics.get("bytes_uploaded", 0) or 0
            )
            publish_metrics["output_bytes_written"] += int(
                file_metrics.get("bytes_written", 0) or 0
            )
            publish_metrics["output_publish_time(s)"] += float(
                file_metrics.get("publish_time(s)", 0.0) or 0.0
            )
            artifact_entry: Dict[str, Any] = {
                "file": artifact_name,
                "published_uri": published_uri,
                "size_bytes": size_bytes,
                "kept_in_run": bool(task.keep_proof_artifacts),
            }
            if retained_path:
                artifact_entry["local_path"] = retained_path
            artifact_metadata[artifact_name] = artifact_entry

        manifest = {
            "request_id": task.request_id,
            "job_id": task.job_id,
            "lease_id": task.lease_id,
            "node_id": task.node_id,
            "setup_key": task.setup_key,
            "setup_dir": task.setup_dir,
            "storage_backend": task.output_backend,
            "s3_bucket": task.output_s3_bucket,
            "srs_path": proof_stages.srs_path,
            "keep_proof_artifacts": bool(task.keep_proof_artifacts),
            "artifacts": artifact_metadata,
            "transfer_metrics": task.transfer_metrics,
            "output_publish_metrics": publish_metrics,
        }
        manifest_path = os.path.join(task.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4)

        published_manifest, manifest_metrics = self._publish_file(
            manifest_path,
            task,
            "manifest.json",
        )
        publish_metrics["artifacts"]["manifest.json"] = manifest_metrics
        publish_metrics["output_bytes_uploaded"] += int(
            manifest_metrics.get("bytes_uploaded", 0) or 0
        )
        publish_metrics["output_bytes_written"] += int(
            manifest_metrics.get("bytes_written", 0) or 0
        )
        publish_metrics["output_publish_time(s)"] += float(
            manifest_metrics.get("publish_time(s)", 0.0) or 0.0
        )

        return ArtifactRef(
            backend=task.output_backend,
            uri=published_manifest,
            s3_bucket=task.output_s3_bucket,
            size_bytes=os.path.getsize(manifest_path),
        ), publish_metrics

    @staticmethod
    def _flatten_transfer_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        model = metrics.get("model", {}) or {}
        input_metrics = metrics.get("input", {}) or {}
        setup = metrics.get("setup", {}) or {}
        return {
            "transfer_model_cache_hit": bool(model.get("cache_hit", False)),
            "transfer_input_cache_hit": bool(input_metrics.get("cache_hit", False)),
            "transfer_model_bytes_downloaded": int(
                model.get("bytes_downloaded", 0) or 0
            ),
            "transfer_input_bytes_downloaded": int(
                input_metrics.get("bytes_downloaded", 0) or 0
            ),
            "transfer_model_bytes_from_local_cache": int(
                model.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_input_bytes_from_local_cache": int(
                input_metrics.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_setup_bytes_downloaded": int(
                setup.get("bytes_downloaded", 0) or 0
            ),
            "transfer_setup_shared_read_only": bool(setup.get("shared_read_only", False)),
            "transfer_setup_bytes_reused_shared": int(setup.get("bytes_reused_shared", 0) or 0),
            "transfer_setup_bytes_from_local_cache": int(
                setup.get("bytes_from_local_cache", 0) or 0
            ),
            "transfer_setup_restore_time(s)": float(
                setup.get("restore_time(s)", 0.0) or 0.0
            ),
            "transfer_pk_decompress_time(s)": float(
                setup.get("pk_decompress_time(s)", 0.0) or 0.0
            ),
        }

    @staticmethod
    def _flatten_output_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output_bytes_uploaded": int(
                metrics.get("output_bytes_uploaded", 0) or 0
            ),
            "output_bytes_written": int(
                metrics.get("output_bytes_written", 0) or 0
            ),
            "output_publish_time(s)": float(
                metrics.get("output_publish_time(s)", 0.0) or 0.0
            ),
        }


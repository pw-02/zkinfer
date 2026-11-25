from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional

from zkinfer.runtime.resource_profiles import update_resource_profile
from zkinfer.storage.io import (
    exists,
    load_json,
    read_csv_as_dict,
    write_dict_to_csv,
)


def _numeric_values(data: Dict[str, Any], key: str) -> list[float]:
    value = data.get(key, [])
    if not isinstance(value, list):
        value = [value]
    values = []
    for item in value:
        try:
            parsed = float(item)
            if math.isfinite(parsed):
                values.append(parsed)
        except (TypeError, ValueError):
            continue
    return values


def _safe_sum(data: Dict[str, Any], key: str) -> float:
    return sum(_numeric_values(data, key))


def _safe_max(data: Dict[str, Any], key: str) -> Optional[float]:
    values = _numeric_values(data, key)
    return max(values) if values else None


def _duration_seconds(start, end) -> Optional[float]:
    if not start or not end:
        return None
    return (end - start).total_seconds()


def _load_existing_resource_profile(job) -> Dict[str, Any]:
    profiling_path = getattr(job, "profiling_file_path", None)
    setup_ref = getattr(job, "setup_cache_ref", None)
    if not profiling_path or setup_ref is None or setup_ref.is_empty:
        return {}

    backend = getattr(setup_ref, "backend", "filesystem") or "filesystem"
    s3_bucket = getattr(setup_ref, "s3_bucket", "") or None
    if not exists(profiling_path, storage_type=backend, s3_bucket=s3_bucket):
        return {}
    return load_json(profiling_path, storage_type=backend, s3_bucket=s3_bucket)


def write_job_report(
    job,
    perf_metrics: Optional[Dict[str, Any]] = None,
    out_dir: str = "reports",
    *, filename: str = "job_report.csv",
) -> Dict[str, Any]:
    """Append one canonical derived row for a completed/failed proof job."""
    perf_metrics = perf_metrics or {}
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "request_id": job.inference_request_id,
        "job_id": job.job_id,
        "lease_id": job.lease_id,
        "job_name": job.job_name,
        "setup_key": perf_metrics.get("setup_key") or getattr(job, "artifact_key", ""),
        "node_id": perf_metrics.get("node_id", job.assigned_node_id or "unknown"),
        "model_path": job.model_path,
        "input_path": job.input_path,
        "job_status": job.job_status.value,
        "message": job.error_message,
        "queued_time": job.queued_time.isoformat() if job.queued_time else None,
        "started_time": job.started_time.isoformat() if job.started_time else None,
        "completed_time": job.completed_time.isoformat() if job.completed_time else None,
        "queue_wait_time(s)": _duration_seconds(job.queued_time, job.started_time),
        "job_runtime(s)": _duration_seconds(job.started_time, job.completed_time),
        "total_elapsed_time(s)": _duration_seconds(job.queued_time, job.completed_time),
        "predicted_duration(s)": job.predicted_duration_s,
        "predicted_peak_memory(GB)": job.predicted_peak_memory_gb,
        "resource_sampling_interval(s)": perf_metrics.get("resource_sampling_interval(s)"),
        "model_write_time(s)": job.model_write_time,
        "input_write_time(s)": job.input_write_time,
        "ezkl_setup_time(s)": perf_metrics.get("ezkl_setup_time(s)"),
        "ezkl_calibrate_settings_time(s)": perf_metrics.get("ezkl_calibrate_settings_time(s)"),
        "ezkl_compile_circuit_time(s)": perf_metrics.get("ezkl_compile_circuit_time(s)"),
        "ezkl_get_srs_time(s)": perf_metrics.get("ezkl_get_srs_time(s)"),
        "ezkl_gen_witness_time(s)": perf_metrics.get("ezkl_gen_witness_time(s)"),
        "ezkl_key_gen_time(s)": perf_metrics.get("ezkl_key_gen_time(s)"),
        "ezkl_proof_time(s)": perf_metrics.get("ezkl_proof_time(s)"),
        "pk_file_size(GB)": perf_metrics.get("pk_file_size(GB)"),
        "vk_file_size(GB)": perf_metrics.get("vk_file_size(GB)"),
        "max_system_memory(GB)": perf_metrics.get("max_system_memory(GB)"),
        "max_process_memory(GB)": perf_metrics.get("max_process_memory(GB)"),
        "avg_system_cpu(%)": perf_metrics.get("avg_system_cpu(%)"),
        "avg_process_cpu_raw(%)": perf_metrics.get("avg_process_cpu_raw(%)"),
        "avg_process_cpu_machine(%)": perf_metrics.get("avg_process_cpu_machine(%)"),
        "fft_count": perf_metrics.get("fft_count"),
        "fft_total_time(s)": perf_metrics.get("fft_total_time(s)"),
        "msm_count": perf_metrics.get("msm_count"),
        "msm_total_time(s)": perf_metrics.get("msm_total_time(s)"),
    }

    for key, value in perf_metrics.items():
        if (
            key.startswith("ezkl_")
            or key.startswith("stage_")
            or key.startswith("transfer_")
            or key.startswith("output_")
            or key.startswith("prover_")
            or key.startswith("node_")
            or key.startswith("halo2_phase_")
        ):
            report[key] = value

    write_dict_to_csv(report, os.path.join(out_dir, filename))

    if filename == "job_report.csv" and report["job_status"] == "COMPLETED" and getattr(job, "profiling_file_path", None):
        existing_profile = _load_existing_resource_profile(job)
        return update_resource_profile(
            existing_profile,
            report,
            setup_key=report["setup_key"],
        )

    return report


def write_failed_attempt_report(job, perf_metrics, out_dir):
    return write_job_report(job, perf_metrics, out_dir, filename="failed_attempt_report.csv")


def write_request_report(request, out_dir: str = "reports") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    job_data = read_csv_as_dict(os.path.join(out_dir, "job_report.csv"))

    preparation_time_s = _duration_seconds(request.created_time, request.queued_time)
    queue_wait_time_s = _duration_seconds(request.queued_time, request.started_time)
    request_runtime_s = _duration_seconds(request.started_time, request.completed_time)
    total_elapsed_time_s = _duration_seconds(request.created_time, request.completed_time)

    report = {
        "request_id": request.request_id,
        "request_name": request.name,
        "onnx_model_path": request.onnx_model_path,
        "input_data_path": request.input_data_path,
        "split_mode": request.split_mode,
        "operators_per_partition": request.ops_per_chunk,
        "scheduler": request.scheduler,
        "num_proof_jobs": len(request.proof_jobs),
        "created_time": request.created_time.isoformat() if request.created_time else None,
        "queued_time": request.queued_time.isoformat() if request.queued_time else None,
        "started_time": request.started_time.isoformat() if request.started_time else None,
        "completed_time": request.completed_time.isoformat() if request.completed_time else None,
        "request_status": request.request_status.value,
        "error_message": request.error_message,
        "preparation_time(s)": preparation_time_s,
        "queue_wait_time(s)": queue_wait_time_s,
        "request_runtime(s)": request_runtime_s,
        "total_elapsed_time(s)": total_elapsed_time_s,
        "verification_time(s)": float(request.preparation_metrics.get("verification_time(s)", 0.0) or 0.0),
        "witness_preparation_mode": request.preparation_metrics.get("witness_preparation_mode", "independent"),
        "preparation_dispatch": request.preparation_metrics.get("preparation_dispatch", "batch"),
        "preparation_cache_enabled": request.preparation_metrics.get("preparation_cache_enabled"),
        "first_job_ready_time(s)": request.preparation_metrics.get("first_job_ready_time(s)"),
        "all_jobs_ready_time(s)": request.preparation_metrics.get("all_jobs_ready_time(s)"),
        "preparation_wall_time(s)": request.preparation_metrics.get("preparation_wall_time(s)"),
        "upfront_preparation_time(s)": request.preparation_metrics.get("upfront_preparation_time(s)", 0.0),
        "upfront_calibration_time(s)": request.preparation_metrics.get("upfront_calibration_time_s", 0.0),
        "upfront_compilation_time(s)": request.preparation_metrics.get("upfront_compilation_time_s", 0.0),
        "upfront_witness_time(s)": request.preparation_metrics.get("upfront_witness_time_s", 0.0),
        "upfront_witness_bytes": request.preparation_metrics.get("upfront_witness_bytes", 0),
        "upfront_compiled_bytes": request.preparation_metrics.get("upfront_compiled_bytes", 0),
        "coordinator_process_max_rss_kib": request.preparation_metrics.get("coordinator_process_max_rss_kib", 0),
        "upfront_setup_cache_hits": request.preparation_metrics.get("upfront_setup_cache_hit", 0),
        "upfront_rescaled_inputs": request.preparation_metrics.get("upfront_rescaled_inputs", 0),
        "agg_prepared_witness_load_time(s)": _safe_sum(job_data, "ezkl_load_prepared_witness_time(s)"),
        "agg_job_runtime(s)": _safe_sum(job_data, "job_runtime(s)"),
        "agg_ezkl_setup_time(s)": _safe_sum(job_data, "ezkl_setup_time(s)"),
        "agg_ezkl_key_gen_time(s)": _safe_sum(job_data, "ezkl_key_gen_time(s)"),
        "agg_ezkl_proof_time(s)": _safe_sum(job_data, "ezkl_proof_time(s)"),
        "agg_stage_admission_wait_time(s)": _safe_sum(
            job_data,
            "stage_admission_wait_time(s)",
        ),
        "agg_fft_time(s)": _safe_sum(job_data, "fft_total_time(s)"),
        "agg_msm_time(s)": _safe_sum(job_data, "msm_total_time(s)"),
        "max_system_memory(GB)": _safe_max(job_data, "max_system_memory(GB)"),
        "max_process_memory(GB)": _safe_max(job_data, "max_process_memory(GB)"),
        "agg_pk_size(GB)": _safe_sum(job_data, "pk_file_size(GB)"),
    }

    for key in job_data:
        if key.startswith("stage_") and key.endswith("_max_memory(GB)"):
            report[f"max_{key}"] = _safe_max(job_data, key)
            stage = key[len("stage_") : -len("_max_memory(GB)")]
            report[f"num_jobs_with_stage_{stage}_memory_samples"] = len(
                _numeric_values(job_data, key)
            )

    write_dict_to_csv(report, os.path.join(out_dir, "request_report.csv"))
    return report

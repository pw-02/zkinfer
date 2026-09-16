import os
from typing import Any, Dict, Optional

from zkinfer.storage.io import read_csv_as_dict, write_dict_to_csv


def _numeric_values(data: Dict[str, Any], key: str):
    value = data.get(key, [])

    if not isinstance(value, list):
        value = [value]

    values = []
    for item in value:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue

    return values


def _safe_sum(data: Dict[str, Any], key: str) -> float:
    return sum(_numeric_values(data, key))


def _safe_max(data: Dict[str, Any], key: str) -> float:
    values = _numeric_values(data, key)
    return max(values) if values else 0.0


def _safe_avg(data: Dict[str, Any], key: str) -> float:
    values = _numeric_values(data, key)
    return sum(values) / len(values) if values else 0.0


def _duration_seconds(start, end) -> Optional[float]:
    if not start or not end:
        return None
    return (end - start).total_seconds()


def write_job_report(
    job,
    perf_metrics: Optional[Dict[str, Any]] = None,
    out_dir: str = "reports",
) -> Dict[str, Any]:
    perf_metrics = perf_metrics or {}

    os.makedirs(out_dir, exist_ok=True)

    queue_wait_time_s = _duration_seconds(job.queued_time, job.started_time)
    job_runtime_s = _duration_seconds(job.started_time, job.completed_time)
    total_elapsed_time_s = _duration_seconds(job.queued_time, job.completed_time)

    total_s3_write_time = (
        perf_metrics.get("ezkl_setup_s3_write_time(s)", 0.0) or 0.0
    )
    total_s3_write_time += job.model_write_time or 0.0
    total_s3_write_time += job.input_write_time or 0.0

    report = {
        "request_id": job.inference_request_id,
        "job_id": job.job_id,
        "job_name": job.job_name,
        "worker_id": perf_metrics.get("worker_id", "unknown"),

        "model_path": job.model_path,
        "input_path": job.input_path,
        "job_status": job.job_status.value,
        "message": job.error_message,

        "queued_time": job.queued_time.isoformat() if job.queued_time else None,
        "started_time": job.started_time.isoformat() if job.started_time else None,
        "completed_time": job.completed_time.isoformat() if job.completed_time else None,

        "queue_wait_time(s)": queue_wait_time_s,
        "job_runtime(s)": job_runtime_s,
        "total_elapsed_time(s)": total_elapsed_time_s,

        "model_write_time(s)": job.model_write_time,
        "input_write_time(s)": job.input_write_time,

        # EZKL timings
        "ezkl_setup_time(s)": perf_metrics.get("ezkl_setup_time(s)", 0.0),
        "ezkl_calibrate_settings_time(s)": perf_metrics.get(
            "ezkl_calibrate_settings_time(s)", 0.0
        ),
        "ezkl_compile_circuit_time(s)": perf_metrics.get(
            "ezkl_compile_circuit_time(s)", 0.0
        ),
        "ezkl_get_srs_time(s)": perf_metrics.get("ezkl_get_srs_time(s)", 0.0),
        "ezkl_gen_witness_time(s)": perf_metrics.get(
            "ezkl_gen_witness_time(s)", 0.0
        ),
        "ezkl_key_gen_time(s)": perf_metrics.get("ezkl_key_gen_time(s)", 0.0),
        "ezkl_proof_time(s)": perf_metrics.get("ezkl_proof_time(s)", 0.0),

        # Storage/cache
        "ezkl_setup_s3_read_time(s)": perf_metrics.get(
            "ezkl_setup_s3_read_time(s)", 0.0
        ),
        "model_s3_read_time(s)": perf_metrics.get(
            "model_s3_read_time(s)", 0.0
        ),
        "input_s3_read_time(s)": perf_metrics.get(
            "input_s3_read_time(s)", 0.0
        ),
        "input_parse_time(s)": perf_metrics.get("input_parse_time(s)", 0.0),
        "input_transfer_bytes": perf_metrics.get("input_transfer_bytes", 0),
        "transfer_only": perf_metrics.get("transfer_only", False),
        "ezkl_setup_s3_write_time(s)": perf_metrics.get(
            "ezkl_setup_s3_write_time(s)", 0.0
        ),
        "total_s3_write_time(s)": total_s3_write_time,

        # Circuit/proving metadata
        "circuit_size(n)": perf_metrics.get("circuit_size(n)", 0),
        "pk_file_size(GB)": perf_metrics.get("pk_file_size(GB)", 0.0),
        "vk_file_size(GB)": perf_metrics.get("vk_file_size(GB)", 0.0),

        # Resource metrics
        "max_system_memory(GB)": perf_metrics.get("max_system_memory(GB)", 0.0),
        "max_process_memory(GB)": perf_metrics.get("max_process_memory(GB)", 0.0),
        "avg_system_cpu(%)": perf_metrics.get("avg_system_cpu(%)", 0.0),
        "avg_process_cpu_raw(%)": perf_metrics.get("avg_process_cpu_raw(%)", 0.0),
        "avg_process_cpu_machine(%)": perf_metrics.get(
            "avg_process_cpu_machine(%)", 0.0
        ),

        # Halo2/EZKL backend metrics
        "fft_count": perf_metrics.get("fft_count", 0),
        "fft_largest": perf_metrics.get("fft_largest", 0),
        "fft_total_time(s)": perf_metrics.get("fft_total_time(s)", 0.0),
        "fft_avg_time(s)": perf_metrics.get("fft_avg_time(s)", 0.0),
        "fft_device": perf_metrics.get("fft_device", "unknown"),

        "msm_count": perf_metrics.get("msm_count", 0),
        "msm_largest": perf_metrics.get("msm_largest", 0),
        "msm_total_time(s)": perf_metrics.get("msm_total_time(s)", 0.0),
        "msm_avg_time(s)": perf_metrics.get("msm_avg_time(s)", 0.0),
        "msm_device": perf_metrics.get("msm_device", "unknown"),
    }

    write_dict_to_csv(report, os.path.join(out_dir, "job_report.csv"))

    if perf_metrics:
        write_dict_to_csv(
            {
                "request_id": job.inference_request_id,
                "job_id": job.job_id,
                **perf_metrics,
            },
            os.path.join(out_dir, "perf_metrics.csv"),
        )

    return report


def write_request_report(
    request,
    out_dir: str = "reports",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    job_report_file = os.path.join(out_dir, "job_report.csv")
    job_data = read_csv_as_dict(job_report_file)

    queue_wait_time_s = _duration_seconds(request.queued_time, request.started_time)
    request_runtime_s = _duration_seconds(request.started_time, request.completed_time)
    total_elapsed_time_s = _duration_seconds(request.created_time, request.completed_time)
    preparation_time_s = _duration_seconds(request.created_time, request.queued_time)
    preparation_pct_of_total = (
        preparation_time_s / total_elapsed_time_s * 100.0
        if preparation_time_s is not None and total_elapsed_time_s
        else None
    )
    max_input_s3_read_time_s = _safe_max(
        job_data,
        "input_s3_read_time(s)",
    )
    aggregate_input_write_time_s = _safe_sum(
        job_data,
        "input_write_time(s)",
    )
    intermediate_generation_time_s = float(
        request.preparation_metrics.get(
            "intermediate_tensor_generation_time(s)",
            0.0,
        )
        or 0.0
    )
    estimated_tensor_communication_time_s = (
        aggregate_input_write_time_s + max_input_s3_read_time_s
    )
    estimated_tensor_overhead_time_s = (
        intermediate_generation_time_s
        + estimated_tensor_communication_time_s
    )
    estimated_tensor_overhead_pct = (
        estimated_tensor_overhead_time_s / total_elapsed_time_s * 100.0
        if total_elapsed_time_s
        else None
    )
    transfer_only = bool(request.preparation_metrics.get("transfer_only", False))
    aggregate_input_transfer_bytes = _safe_sum(
        job_data,
        "input_transfer_bytes",
    )
    max_input_transfer_bytes = _safe_max(
        job_data,
        "input_transfer_bytes",
    )
    boundary_microbenchmark_wall_time_s = (
        intermediate_generation_time_s
        + aggregate_input_write_time_s
        + (request_runtime_s or 0.0)
        if transfer_only
        else None
    )

    report = {
        "request_id": request.request_id,
        "request_name": request.name,

        "onnx_model_path": request.onnx_model_path,
        "input_data_path": request.input_data_path,
        "split_mode": request.split_mode,
        "ops_per_chunk": request.ops_per_chunk,
        "scheduler": request.scheduler,

        "num_proof_jobs": len(request.proof_jobs),

        "created_time": (
            request.created_time.isoformat()
            if request.created_time
            else None
        ),
        "queued_time": (
            request.queued_time.isoformat()
            if request.queued_time
            else None
        ),
        "started_time": (
            request.started_time.isoformat()
            if request.started_time
            else None
        ),
        "completed_time": (
            request.completed_time.isoformat()
            if request.completed_time
            else None
        ),

        "request_status": request.request_status.value,
        "error_message": request.error_message,

        "queue_wait_time(s)": queue_wait_time_s,
        "preparation_time(s)": preparation_time_s,
        "preparation_time_pct_of_total(%)": preparation_pct_of_total,
        "request_runtime(s)": request_runtime_s,
        "total_elapsed_time(s)": total_elapsed_time_s,
        "transfer_only": transfer_only,

        # Aggregated job timings
        "agg_job_runtime(s)": _safe_sum(job_data, "job_runtime(s)"),
        "agg_ezkl_setup_time(s)": _safe_sum(job_data, "ezkl_setup_time(s)"),
        "agg_ezkl_calibrate_settings_time(s)": _safe_sum(
            job_data, "ezkl_calibrate_settings_time(s)"
        ),
        "agg_ezkl_compile_circuit_time(s)": _safe_sum(
            job_data, "ezkl_compile_circuit_time(s)"
        ),
        "agg_ezkl_get_srs_time(s)": _safe_sum(job_data, "ezkl_get_srs_time(s)"),
        "agg_ezkl_gen_witness_time(s)": _safe_sum(
            job_data, "ezkl_gen_witness_time(s)"
        ),
        "agg_ezkl_key_gen_time(s)": _safe_sum(job_data, "ezkl_key_gen_time(s)"),
        "agg_ezkl_proof_time(s)": _safe_sum(job_data, "ezkl_proof_time(s)"),

        # Aggregated backend metrics
        "agg_fft_time(s)": _safe_sum(job_data, "fft_total_time(s)"),
        "agg_msm_time(s)": _safe_sum(job_data, "msm_total_time(s)"),
        "agg_circuit_size(n)": _safe_sum(job_data, "circuit_size(n)"),

        # Resource summaries
        "max_system_memory(GB)": _safe_max(job_data, "max_system_memory(GB)"),
        "max_process_memory(GB)": _safe_max(job_data, "max_process_memory(GB)"),
        "avg_system_cpu(%)": _safe_avg(job_data, "avg_system_cpu(%)"),
        "avg_process_cpu_raw(%)": _safe_avg(job_data, "avg_process_cpu_raw(%)"),
        "avg_process_cpu_machine(%)": _safe_avg(
            job_data,
            "avg_process_cpu_machine(%)",
        ),

        # Artifact sizes
        "max_pk_file_size(GB)": _safe_max(job_data, "pk_file_size(GB)"),
        "max_vk_file_size(GB)": _safe_max(job_data, "vk_file_size(GB)"),

        # Storage
        "agg_model_write_time(s)": _safe_sum(job_data, "model_write_time(s)"),
        "agg_input_write_time(s)": aggregate_input_write_time_s,
        "agg_model_s3_read_time(s)": _safe_sum(
            job_data, "model_s3_read_time(s)"
        ),
        "agg_input_s3_read_time(s)": _safe_sum(
            job_data, "input_s3_read_time(s)"
        ),
        "agg_s3_read_time(s)": (
            _safe_sum(job_data, "ezkl_setup_s3_read_time(s)")
            + _safe_sum(job_data, "model_s3_read_time(s)")
            + _safe_sum(job_data, "input_s3_read_time(s)")
        ),
        "agg_s3_write_time(s)": _safe_sum(job_data, "total_s3_write_time(s)"),
        "max_input_s3_read_time(s)": max_input_s3_read_time_s,
        "agg_input_transfer_bytes": aggregate_input_transfer_bytes,
        "agg_input_transfer_MiB": aggregate_input_transfer_bytes / (1024 ** 2),
        "max_job_input_transfer_bytes": max_input_transfer_bytes,
        "max_job_input_transfer_MiB": max_input_transfer_bytes / (1024 ** 2),
        "boundary_microbenchmark_wall_time(s)": (
            boundary_microbenchmark_wall_time_s
        ),
        "estimated_tensor_communication_time(s)": (
            estimated_tensor_communication_time_s
        ),
        "estimated_tensor_overhead_time(s)": estimated_tensor_overhead_time_s,
        "estimated_tensor_overhead_pct_of_total(%)": (
            estimated_tensor_overhead_pct
        ),
        **request.preparation_metrics,
    }

    write_dict_to_csv(report, os.path.join(out_dir, "request_report.csv"))
    return report

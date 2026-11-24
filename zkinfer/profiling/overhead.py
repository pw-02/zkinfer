from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable


SUM_FIELDS = [
    "transfer_model_bytes_downloaded",
    "transfer_input_bytes_downloaded",
    "transfer_setup_bytes_downloaded",
    "transfer_model_bytes_from_local_cache",
    "transfer_input_bytes_from_local_cache",
    "transfer_setup_bytes_from_local_cache",
    "transfer_setup_bytes_uploaded",
    "output_bytes_uploaded",
    "output_bytes_written",
    "model_materialize_time(s)",
    "input_materialize_time(s)",
    "transfer_setup_restore_time(s)",
    "transfer_pk_decompress_time(s)",
    "transfer_setup_publish_time(s)",
    "transfer_pk_compress_time(s)",
    "output_publish_time(s)",
]

CACHE_HIT_FIELDS = [
    "transfer_model_cache_hit",
    "transfer_input_cache_hit",
]


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_job_report(job_report_csv: str) -> list[Dict[str, str]]:
    if not os.path.exists(job_report_csv):
        return []
    with open(job_report_csv, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def aggregate_transfer_overhead(rows: Iterable[Dict[str, str]]) -> Dict[str, float | int]:
    rows = list(rows)
    report: Dict[str, float | int] = {"num_jobs": len(rows)}

    for field in SUM_FIELDS:
        report[f"agg_{field}"] = sum(_to_float(row.get(field)) for row in rows)

    for field in CACHE_HIT_FIELDS:
        hits = sum(1 for row in rows if _to_bool(row.get(field)))
        report[f"{field}_count"] = hits
        report[f"{field}_rate"] = hits / len(rows) if rows else 0.0

    total_downloaded = (
        report.get("agg_transfer_model_bytes_downloaded", 0.0)
        + report.get("agg_transfer_input_bytes_downloaded", 0.0)
        + report.get("agg_transfer_setup_bytes_downloaded", 0.0)
    )
    total_local = (
        report.get("agg_transfer_model_bytes_from_local_cache", 0.0)
        + report.get("agg_transfer_input_bytes_from_local_cache", 0.0)
        + report.get("agg_transfer_setup_bytes_from_local_cache", 0.0)
    )
    total_uploaded = (
        report.get("agg_transfer_setup_bytes_uploaded", 0.0)
        + report.get("agg_output_bytes_uploaded", 0.0)
    )

    report["agg_bytes_downloaded_from_shared_store"] = total_downloaded
    report["agg_bytes_served_from_node_cache"] = total_local
    report["agg_bytes_uploaded_to_shared_store"] = total_uploaded
    report["agg_downloaded_MiB"] = total_downloaded / (1024**2)
    report["agg_node_cache_served_MiB"] = total_local / (1024**2)
    report["agg_uploaded_MiB"] = total_uploaded / (1024**2)
    report["node_cache_byte_hit_rate"] = (
        total_local / (total_local + total_downloaded)
        if (total_local + total_downloaded)
        else 0.0
    )

    report["agg_data_movement_service_time(s)"] = (
        report.get("agg_model_materialize_time(s)", 0.0)
        + report.get("agg_input_materialize_time(s)", 0.0)
        + report.get("agg_transfer_setup_restore_time(s)", 0.0)
        + report.get("agg_transfer_setup_publish_time(s)", 0.0)
        + report.get("agg_output_publish_time(s)", 0.0)
    )
    return report


def write_transfer_overhead_report(out_dir: str) -> Dict[str, float | int]:
    job_report_csv = os.path.join(out_dir, "job_report.csv")
    rows = read_job_report(job_report_csv)
    report = aggregate_transfer_overhead(rows)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = Path(out_dir) / "transfer_overhead_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)
    return report

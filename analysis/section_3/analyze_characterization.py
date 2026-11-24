#!/usr/bin/env python3
"""Generate Section 3 characterization tables and figures from zkInfer runs.

The analysis is policy-independent. It consumes raw resource traces and exact
stage labels and quantifies stage demand, repeatability, reservation
opportunity, historical memory prediction, cold versus cached setup, and node
activity under concurrent requests.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .trace_layout import find_job_dirs, job_identity
except ImportError:
    from trace_layout import find_job_dirs, job_identity


PROCESS_TYPES = {"process", "process_tree"}
IGNORE_STAGES = {"", "UNKNOWN", "STARTING", "DONE", "FAILED"}
PREDICTED_MEMORY_RE = re.compile(
    r"^stage_admission_(?P<stage>.+)_predicted_memory\(GB\)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/section_3/generated"),
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.0,
        help=(
            "Usable node memory for pairwise packing analysis after the fixed "
            "node memory safety reserve."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _metric_stage(stage: str) -> str:
    stage = (stage or "UNKNOWN").strip().lower()
    stage = re.sub(r"[^a-z0-9]+", "_", stage).strip("_")
    return stage or "unknown"


def _ezkl_stage(stage: str) -> str:
    stage = (stage or "UNKNOWN").strip()
    if stage.startswith("PROVING::"):
        return "PROVING"
    if stage.startswith("WAITING_FOR_"):
        return stage
    return stage


def _workload_from_request_id(request_id: str) -> str:
    marker = "_split-"
    return request_id.split(marker, 1)[0] if marker in request_id else request_id


def _safe_cv(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    if len(series) < 2:
        return float("nan")
    mean = float(series.mean())
    if mean == 0:
        return float("nan")
    return float(series.std(ddof=1) / mean)


def _weighted_duration(samples: pd.DataFrame) -> pd.Series:
    elapsed = pd.to_numeric(samples["elapsed_s"], errors="coerce")
    deltas = elapsed.shift(-1) - elapsed
    positive = deltas[deltas > 0]
    fallback = float(positive.median()) if not positive.empty else 0.0
    return deltas.where(deltas > 0, fallback).fillna(fallback)


def _find_job_dirs(runs_dir: Path) -> list[Path]:
    return find_job_dirs(runs_dir)


def _load_job_reports(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for path in runs_dir.glob("*/reports/job_report.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "job_id" not in frame.columns:
            continue
        frame["_job_report_source"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    return data.drop_duplicates(subset=["request_id", "job_id"], keep="last")


def _job_report_row(
    reports: pd.DataFrame,
    request_id: str,
    job_id: str,
) -> Dict[str, Any]:
    if reports.empty:
        return {}
    rows = reports[
        (reports["request_id"] == request_id) & (reports["job_id"] == job_id)
    ]
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def _read_setup_key(job_dir: Path, report_row: Dict[str, Any]) -> str:
    value = report_row.get("setup_key")
    if value is not None and str(value).strip() and str(value).lower() != "nan":
        return str(value)
    path = job_dir / "circuit_features.json"
    if not path.is_file():
        path = job_dir / "admission_profile_before.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("setup_key") or "")
    except Exception:
        return ""


def _cache_state(report_row: Dict[str, Any]) -> str:
    cached = _as_bool(report_row.get("ezkl_key_gen_used_cache"))
    if cached is True:
        return "cached_setup"
    if cached is False:
        return "cold_setup"
    return "unknown"


def _resource_frame(job_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(job_dir / "resource_usage.csv")
    for column in (
        "elapsed_s",
        "memory_gb",
        "cpu_percent",
        "cpu_machine_percent",
        "node_active_jobs",
        "node_reserved_memory_gb",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _job_identity(job_dir: Path) -> tuple[str, str, str]:
    request_id, job_id = job_identity(job_dir)
    return request_id, job_id, _workload_from_request_id(request_id)


def _artifact_validation(job_dir: Path) -> Dict[str, Any]:
    request_id, job_id, workload = _job_identity(job_dir)
    names = [
        "resource_usage.csv",
        "execution_trace.csv",
        "stage_transitions.csv",
        "halo2_stage_transitions.csv",
        "halo2_prover_cpu.csv",
        "admission_profile_before.json",
        "status.txt",
    ]
    result: Dict[str, Any] = {
        "request_id": request_id,
        "job_id": job_id,
        "workload": workload,
        "job_dir": str(job_dir),
    }
    for name in names:
        result[f"has_{name.replace('.', '_')}"] = (job_dir / name).exists()
    result["complete_for_stage_analysis"] = bool(
        result["has_resource_usage_csv"] and (result["has_stage_transitions_csv"] or result["has_execution_trace_csv"])
    )
    return result


def _stage_rows(
    resource: pd.DataFrame,
    request_id: str,
    job_id: str,
    workload: str,
    setup_key: str,
    cache_state: str,
) -> list[Dict[str, Any]]:
    process = resource[resource["type"].isin(PROCESS_TYPES)].copy()
    if process.empty:
        return []
    process = process.sort_values("elapsed_s")
    process["sample_duration_s"] = _weighted_duration(process)
    process["stage"] = process["stage"].fillna("UNKNOWN").astype(str)

    rows = []
    for stage, group in process.groupby("stage", sort=False):
        if stage in IGNORE_STAGES:
            continue
        rows.append(
            {
                "request_id": request_id,
                "job_id": job_id,
                "workload": workload,
                "setup_key": setup_key,
                "cache_state": cache_state,
                "stage": stage,
                "ezkl_stage": _ezkl_stage(stage),
                "is_halo2_subphase": stage.startswith("PROVING::"),
                "duration_s": float(group["sample_duration_s"].sum()),
                "peak_memory_gb": float(group["memory_gb"].max()),
                "mean_memory_gb": float(group["memory_gb"].mean()),
                "mean_cpu_machine_percent": float(group["cpu_machine_percent"].mean()),
                "samples": int(len(group)),
            }
        )
    return rows


def _reservation_rows(
    resource: pd.DataFrame,
    request_id: str,
    job_id: str,
    workload: str,
    setup_key: str,
    cache_state: str,
) -> list[Dict[str, Any]]:
    process = resource[resource["type"].isin(PROCESS_TYPES)].copy()
    if process.empty:
        return []
    process = process.sort_values("elapsed_s")
    process["stage"] = process["stage"].fillna("UNKNOWN").astype(str)
    process = process[~process["stage"].isin(IGNORE_STAGES)].copy()
    if process.empty:
        return []
    process["dt"] = _weighted_duration(process)
    actual = process["memory_gb"].fillna(0.0).clip(lower=0.0)
    whole_peak = float(actual.max())

    exact_peak = process.groupby("stage")["memory_gb"].max().to_dict()
    process["ezkl_stage"] = process["stage"].map(_ezkl_stage)
    ezkl_peak = process.groupby("ezkl_stage")["memory_gb"].max().to_dict()

    reservations = {
        "whole_job": pd.Series(whole_peak, index=process.index, dtype="float64"),
        "ezkl_stage": process["ezkl_stage"].map(ezkl_peak).astype(float),
        "halo2_subphase": process["stage"].map(exact_peak).astype(float),
    }

    rows = []
    for granularity, reserved in reservations.items():
        reserved = reserved.fillna(whole_peak).clip(lower=0.0)
        unused = (reserved - actual).clip(lower=0.0)
        reserved_gb_s = float((reserved * process["dt"]).sum())
        unused_gb_s = float((unused * process["dt"]).sum())
        total_s = float(process["dt"].sum())
        rows.append(
            {
                "request_id": request_id,
                "job_id": job_id,
                "workload": workload,
                "setup_key": setup_key,
                "cache_state": cache_state,
                "granularity": granularity,
                "duration_s": total_s,
                "whole_job_peak_memory_gb": whole_peak,
                "average_reserved_memory_gb": (
                    reserved_gb_s / total_s if total_s > 0 else 0.0
                ),
                "average_actual_memory_gb": (
                    float((actual * process["dt"]).sum()) / total_s
                    if total_s > 0
                    else 0.0
                ),
                "reserved_gb_seconds": reserved_gb_s,
                "unused_gb_seconds": unused_gb_s,
                "normalized_reservation_waste": (
                    unused_gb_s / reserved_gb_s if reserved_gb_s > 0 else 0.0
                ),
            }
        )
    return rows


def _job_row(
    resource: pd.DataFrame,
    request_id: str,
    job_id: str,
    workload: str,
    setup_key: str,
    cache_state: str,
    report_row: Dict[str, Any],
) -> Dict[str, Any]:
    process = resource[resource["type"].isin(PROCESS_TYPES)].copy()
    system = resource[resource["type"] == "system"].copy()
    return {
        "request_id": request_id,
        "job_id": job_id,
        "workload": workload,
        "setup_key": setup_key,
        "cache_state": cache_state,
        "peak_process_memory_gb": float(process["memory_gb"].max())
        if not process.empty else float("nan"),
        "peak_system_memory_gb": float(system["memory_gb"].max())
        if not system.empty else float("nan"),
        "mean_process_cpu_machine_percent": float(
            process["cpu_machine_percent"].mean()
        ) if not process.empty else float("nan"),
        "observed_trace_duration_s": float(process["elapsed_s"].max())
        if not process.empty else 0.0,
        "reported_job_runtime_s": _as_float(
            report_row.get("job_runtime(s)"), float("nan")
        ),
        "ezkl_setup_time_s": _as_float(
            report_row.get("ezkl_setup_time(s)"), float("nan")
        ),
        "ezkl_witness_time_s": _as_float(
            report_row.get("ezkl_gen_witness_time(s)"), float("nan")
        ),
        "ezkl_keygen_time_s": _as_float(
            report_row.get("ezkl_key_gen_time(s)"), float("nan")
        ),
        "ezkl_proof_time_s": _as_float(
            report_row.get("ezkl_proof_time(s)"), float("nan")
        ),
        "pk_file_size_gb": _as_float(
            report_row.get("pk_file_size(GB)"), float("nan")
        ),
    }


def _prediction_rows(
    report_row: Dict[str, Any],
    stage_rows: list[Dict[str, Any]],
    request_id: str,
    job_id: str,
    workload: str,
    setup_key: str,
    cache_state: str,
) -> list[Dict[str, Any]]:
    actual_by_metric_stage: Dict[str, float] = {}
    for row in stage_rows:
        stage_key = _metric_stage(row["stage"])
        actual_by_metric_stage[stage_key] = max(
            actual_by_metric_stage.get(stage_key, 0.0),
            _as_float(row["peak_memory_gb"]),
        )

    result = []
    for key, value in report_row.items():
        match = PREDICTED_MEMORY_RE.match(str(key))
        if not match:
            continue
        stage = match.group("stage")
        predicted = _as_float(value, float("nan"))
        if not math.isfinite(predicted):
            continue
        if stage == "fused_keygen_prove":
            actual = max(
                actual_by_metric_stage.get("key_gen", 0.0),
                actual_by_metric_stage.get("proving", 0.0),
            )
        else:
            actual = actual_by_metric_stage.get(stage, 0.0)
        if actual <= 0:
            continue
        error = predicted - actual
        result.append(
            {
                "request_id": request_id,
                "job_id": job_id,
                "workload": workload,
                "setup_key": setup_key,
                "cache_state": cache_state,
                "stage": stage,
                "predicted_memory_gb": predicted,
                "actual_memory_gb": actual,
                "error_gb": error,
                "absolute_percent_error": abs(error) / actual * 100.0,
                "underpredicted": error < 0,
                "underprediction_gb": max(0.0, -error),
            }
        )
    return result


def _repeatability(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()
    rows = []
    for (workload, setup_key, stage), group in stage_df.groupby(
        ["workload", "setup_key", "stage"], dropna=False
    ):
        if not setup_key or len(group) < 2:
            continue
        memory = group["peak_memory_gb"].astype(float)
        cpu = group["mean_cpu_machine_percent"].astype(float)
        duration = group["duration_s"].astype(float)
        rows.append(
            {
                "workload": workload,
                "setup_key": setup_key,
                "stage": stage,
                "runs": int(len(group)),
                "memory_mean_gb": float(memory.mean()),
                "memory_cv": _safe_cv(memory),
                "memory_p90_gb": float(memory.quantile(0.90)),
                "memory_max_gb": float(memory.max()),
                "cpu_mean_percent": float(cpu.mean()),
                "cpu_cv": _safe_cv(cpu),
                "duration_mean_s": float(duration.mean()),
                "duration_cv": _safe_cv(duration),
                "duration_p90_s": float(duration.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _cold_warm(job_df: pd.DataFrame) -> pd.DataFrame:
    if job_df.empty:
        return pd.DataFrame()
    valid = job_df[job_df["cache_state"].isin({"cold_setup", "cached_setup"})]
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for (workload, state), group in valid.groupby(["workload", "cache_state"]):
        rows.append(
            {
                "workload": workload,
                "cache_state": state,
                "jobs": int(len(group)),
                "median_peak_process_memory_gb": float(
                    group["peak_process_memory_gb"].median()
                ),
                "median_job_runtime_s": float(group["reported_job_runtime_s"].median()),
                "median_setup_time_s": float(group["ezkl_setup_time_s"].median()),
                "median_keygen_time_s": float(group["ezkl_keygen_time_s"].median()),
                "median_proof_time_s": float(group["ezkl_proof_time_s"].median()),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_packing(
    stage_df: pd.DataFrame,
    job_df: pd.DataFrame,
    memory_budget_gb: float,
) -> pd.DataFrame:
    if memory_budget_gb <= 0 or stage_df.empty or job_df.empty:
        return pd.DataFrame()

    peak_by_job = job_df.set_index(["request_id", "job_id"])[
        "peak_process_memory_gb"
    ].to_dict()
    stage_peaks: Dict[tuple[str, str], list[float]] = {}
    for key, group in stage_df.groupby(["request_id", "job_id"]):
        values = [
            float(value)
            for value in group["peak_memory_gb"].dropna().tolist()
            if float(value) > 0
        ]
        if values:
            stage_peaks[key] = values

    keys = sorted(set(stage_peaks) & set(peak_by_job))
    rows = []
    for left, right in combinations(keys, 2):
        whole_sum = _as_float(peak_by_job[left]) + _as_float(peak_by_job[right])
        if whole_sum <= memory_budget_gb:
            continue
        combinations_fit = [
            a + b <= memory_budget_gb
            for a in stage_peaks[left]
            for b in stage_peaks[right]
        ]
        if not combinations_fit:
            continue
        rows.append(
            {
                "left_request_id": left[0],
                "left_job_id": left[1],
                "right_request_id": right[0],
                "right_job_id": right[1],
                "memory_budget_gb": memory_budget_gb,
                "whole_job_peak_sum_gb": whole_sum,
                "whole_job_blocks_overlap": True,
                "stage_pair_fit_fraction": float(np.mean(combinations_fit)),
                "any_stage_pair_fits": bool(any(combinations_fit)),
            }
        )
    return pd.DataFrame(rows)


def _node_activity(resource_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in resource_frames:
        if frame.empty or "timestamp" not in frame.columns:
            continue
        system = frame[frame["type"] == "system"].copy()
        if system.empty or "node_active_jobs" not in system.columns:
            continue
        system["timestamp"] = pd.to_datetime(system["timestamp"], errors="coerce")
        system = system.dropna(subset=["timestamp"])
        rows.append(
            system[
                [
                    "timestamp",
                    "node_active_jobs",
                    "node_reserved_memory_gb",
                    "memory_gb",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values("timestamp").set_index("timestamp")
    data = data.resample("250ms").max().dropna(how="all").reset_index()
    if not data.empty:
        data["elapsed_s"] = (
            data["timestamp"] - data["timestamp"].min()
        ).dt.total_seconds()
    return data


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _boxplot(
    frame: pd.DataFrame,
    category: str,
    value: str,
    output: Path,
    ylabel: str,
    title: str,
    limit_categories: Optional[int] = None,
) -> None:
    data = frame[[category, value]].dropna()
    if data.empty:
        return
    counts = data.groupby(category)[value].count().sort_values(ascending=False)
    categories = list(counts.index)
    if limit_categories:
        categories = categories[:limit_categories]
    series = [
        data.loc[data[category] == item, value].astype(float).values
        for item in categories
    ]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(categories)), 4.5))
    ax.boxplot(series, tick_labels=categories, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _render_figures(
    out_dir: Path,
    stage_df: pd.DataFrame,
    repeatability: pd.DataFrame,
    reservation: pd.DataFrame,
    prediction: pd.DataFrame,
    activity: pd.DataFrame,
) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ezkl = stage_df[~stage_df["stage"].str.startswith("PROVING::", na=False)].copy()
    _boxplot(
        ezkl,
        "ezkl_stage",
        "peak_memory_gb",
        figures / "fig_ezkl_stage_memory.png",
        "Peak process memory (GB)",
        "Peak memory by EZKL stage",
    )
    _boxplot(
        ezkl,
        "ezkl_stage",
        "duration_s",
        figures / "fig_ezkl_stage_duration.png",
        "Observed stage duration (s)",
        "Duration by EZKL stage",
    )

    halo2 = stage_df[stage_df["stage"].str.startswith("PROVING::", na=False)].copy()
    _boxplot(
        halo2,
        "stage",
        "peak_memory_gb",
        figures / "fig_halo2_subphase_memory.png",
        "Peak process memory (GB)",
        "Peak memory by Halo2 proving subphase",
        limit_categories=15,
    )

    if not reservation.empty:
        summary = reservation.groupby("granularity")[
            "normalized_reservation_waste"
        ].median()
        order = [
            item
            for item in ["whole_job", "ezkl_stage", "halo2_subphase"]
            if item in summary
        ]
        if order:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.bar(order, [summary[item] * 100.0 for item in order])
            ax.set_ylabel("Median reserved-but-unused memory (%)")
            ax.set_title("Reservation waste by accounting granularity")
            ax.tick_params(axis="x", labelrotation=20)
            fig.tight_layout()
            fig.savefig(figures / "fig_reservation_waste.png", dpi=200)
            plt.close(fig)

    if not repeatability.empty:
        plot = repeatability.dropna(subset=["memory_cv"]).copy()
        if not plot.empty:
            grouped = plot.groupby("stage")["memory_cv"].median().sort_values().tail(15)
            fig, ax = plt.subplots(
                figsize=(7.5, max(4.0, 0.3 * len(grouped)))
            )
            ax.barh(grouped.index, grouped.values * 100.0)
            ax.set_xlabel("Median memory CV across setup keys (%)")
            ax.set_title("Repeatability of stage peak memory")
            fig.tight_layout()
            fig.savefig(figures / "fig_repeatability_memory.png", dpi=200)
            plt.close(fig)

    if not prediction.empty:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(
            prediction["actual_memory_gb"],
            prediction["predicted_memory_gb"],
            alpha=0.6,
        )
        upper = max(
            float(prediction["actual_memory_gb"].max()),
            float(prediction["predicted_memory_gb"].max()),
        )
        ax.plot([0, upper], [0, upper], linestyle="--")
        ax.set_xlabel("Measured stage peak memory (GB)")
        ax.set_ylabel("Predicted stage memory (GB)")
        ax.set_title("Historical memory prediction accuracy")
        fig.tight_layout()
        fig.savefig(figures / "fig_prediction_accuracy.png", dpi=200)
        plt.close(fig)

    if not activity.empty and "node_active_jobs" in activity:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.plot(activity["elapsed_s"], activity["node_active_jobs"])
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Active proof jobs")
        ax.set_title("Observed node concurrency")
        fig.tight_layout()
        fig.savefig(figures / "fig_node_activity.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    job_dirs = _find_job_dirs(runs_dir)
    if not job_dirs:
        raise SystemExit(f"No resource_usage.csv files found under {runs_dir}")

    reports = _load_job_reports(runs_dir)
    validation_rows = []
    job_rows = []
    stage_rows = []
    reservation_rows = []
    prediction_rows = []
    resource_frames = []

    for job_dir in job_dirs:
        validation_rows.append(_artifact_validation(job_dir))
        request_id, job_id, workload = _job_identity(job_dir)
        report_row = _job_report_row(reports, request_id, job_id)
        setup_key = _read_setup_key(job_dir, report_row)
        cache_state = _cache_state(report_row)

        try:
            resource = _resource_frame(job_dir)
        except Exception as exc:
            print(f"Skipping unreadable trace {job_dir}: {exc}")
            continue
        resource_frames.append(resource)

        current_stage_rows = _stage_rows(
            resource,
            request_id,
            job_id,
            workload,
            setup_key,
            cache_state,
        )
        stage_rows.extend(current_stage_rows)
        reservation_rows.extend(
            _reservation_rows(
                resource,
                request_id,
                job_id,
                workload,
                setup_key,
                cache_state,
            )
        )
        job_rows.append(
            _job_row(
                resource,
                request_id,
                job_id,
                workload,
                setup_key,
                cache_state,
                report_row,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                report_row,
                current_stage_rows,
                request_id,
                job_id,
                workload,
                setup_key,
                cache_state,
            )
        )

    validation = pd.DataFrame(validation_rows)
    jobs = pd.DataFrame(job_rows)
    stages = pd.DataFrame(stage_rows)
    reservation = pd.DataFrame(reservation_rows)
    prediction = pd.DataFrame(prediction_rows)
    repeatability = _repeatability(stages)
    cold_warm = _cold_warm(jobs)
    pairwise = _pairwise_packing(stages, jobs, args.memory_budget_gb)
    activity = _node_activity(resource_frames)

    _write_csv(validation, out_dir / "artifact_validation.csv")
    _write_csv(jobs, out_dir / "job_characterization.csv")
    _write_csv(stages, out_dir / "stage_characterization.csv")
    _write_csv(repeatability, out_dir / "repeatability.csv")
    _write_csv(reservation, out_dir / "reservation_opportunity.csv")
    _write_csv(prediction, out_dir / "prediction_accuracy.csv")
    _write_csv(pairwise, out_dir / "pairwise_packing.csv")
    _write_csv(cold_warm, out_dir / "cold_warm_summary.csv")
    _write_csv(activity, out_dir / "node_activity.csv")

    summary: Dict[str, Any] = {
        "runs_dir": str(runs_dir),
        "jobs_discovered": int(len(job_dirs)),
        "jobs_analyzed": int(len(jobs)),
        "stages_analyzed": int(len(stages)),
        "setup_keys": int(stages["setup_key"].nunique()) if not stages.empty else 0,
        "memory_budget_gb": args.memory_budget_gb,
    }
    if not reservation.empty:
        summary["median_reservation_waste"] = {
            str(key): float(value)
            for key, value in reservation.groupby("granularity")[
                "normalized_reservation_waste"
            ].median().items()
        }
    if not prediction.empty:
        summary["prediction"] = {
            "observations": int(len(prediction)),
            "median_absolute_percent_error": float(
                prediction["absolute_percent_error"].median()
            ),
            "underprediction_rate": float(prediction["underpredicted"].mean()),
            "p95_underprediction_gb": float(
                prediction["underprediction_gb"].quantile(0.95)
            ),
        }
    if not pairwise.empty:
        summary["pairwise_packing"] = {
            "whole_job_blocked_pairs": int(len(pairwise)),
            "pairs_with_finer_grained_fit": int(pairwise["any_stage_pair_fits"].sum()),
        }
    if not activity.empty:
        summary["node_activity"] = {
            "peak_active_jobs": int(activity["node_active_jobs"].max()),
            "fraction_time_two_or_more_active": float(
                (activity["node_active_jobs"] >= 2).mean()
            ),
        }

    (out_dir / "section3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.no_figures and not stages.empty:
        _render_figures(
            out_dir,
            stages,
            repeatability,
            reservation,
            prediction,
            activity,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote Section 3 analysis to {out_dir}")


if __name__ == "__main__":
    main()

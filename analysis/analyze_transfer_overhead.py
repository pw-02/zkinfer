

#!/usr/bin/env python3
"""Create the small table needed for the partition-overhead rebuttal.

Output columns:
    g, jobs, measured E2E, added tensor overhead, overhead/E2E,
    intermediate-input traffic, transfer-only repetitions.

Inputs may be run directories, request_report.csv files, or ZIP archives.
Repeated runs are summarized using medians.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPORT_SUFFIX = "request_report.csv"
ECECSV_SUFFIX = "transfer_only/example_reference_metrics.csv"
INPUTR_DIRS = ["transfer_only/nano_gpt_4_layers_64"]
NUM_WORKERS = 15
OUTPUT_CSV = "transfer_only/nano_gpt_4_layers_64/transfer_overhead_summary.csv"
MD_REPORT = "transfer_only/nano_gpt_4_layers_64/transfer_overhead_summary.md"

REPORT_NAME = "request_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize transfer-only overhead for the reviewer table."
    )
    parser.add_argument(
        "--inputs",
        default=INPUTR_DIRS,
        nargs="+",
        help="Transfer-only run directories, report CSVs, or ZIP archives.",
    )
    parser.add_argument(
        "--e2e-csv",
        default=None,
        help="Optional CSV with columns model,g,e2e_s. Repeats are allowed.",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=MD_REPORT,
        help="Output Markdown path.",
    )
    return parser.parse_args()


def normalize_model(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "unknown"


def normalize_g(value: object, split_mode: object = None) -> str:
    if str(split_mode or "").strip().lower() == "none":
        return "full"
    text = str(value or "").strip().lower()
    if text in {"full", "none", "|v|", "v", "unsplit"}:
        return "full"
    return str(int(float(text)))


def g_sort_key(g: str) -> Tuple[int, float]:
    return (1, math.inf) if g == "full" else (0, float(g))


def number(value: object) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def read_rows_from_text(text: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def iter_report_rows(path: Path) -> Iterable[Dict[str, str]]:
    if path.is_dir():
        for report in sorted(path.rglob(REPORT_NAME)):
            with report.open("r", encoding="utf-8-sig", newline="") as file:
                yield from csv.DictReader(file)
        return

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(REPORT_NAME):
                    text = archive.read(name).decode("utf-8-sig")
                    yield from read_rows_from_text(text)
        return

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            yield from csv.DictReader(file)
        return

    raise ValueError(f"Unsupported input: {path}")


def boundary_wall_time(row: Dict[str, str]) -> Optional[float]:
    measured = number(row.get("boundary_microbenchmark_wall_time(s)"))
    if measured is not None:
        return measured

    generation = number(row.get("intermediate_tensor_generation_time(s)"))
    upload = number(row.get("input_upload_wall_time(s)"))
    worker_phase = number(row.get("request_runtime(s)"))
    if generation is None or upload is None or worker_phase is None:
        return None
    return generation + upload + worker_phase


def load_transfer_results(
    inputs: Sequence[str],
) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

    for raw_path in inputs:
        for row in iter_report_rows(Path(raw_path)):
            if not truthy(row.get("transfer_only")):
                continue
            if str(row.get("request_status", "")).upper() != "COMPLETED":
                continue

            wall_time = boundary_wall_time(row)
            if wall_time is None:
                continue

            label = row.get("request_name") or row.get("request_id") or "unknown"
            model = normalize_model(label)
            g = normalize_g(row.get("ops_per_chunk"), row.get("split_mode"))

            grouped[(model, g)].append(
                {
                    "label": str(label),
                    "jobs": number(row.get("num_proof_jobs")),
                    "boundary_s": wall_time,
                    "input_mib": number(row.get("agg_input_transfer_MiB")),
                }
            )

    if not grouped:
        raise ValueError("No completed transfer-only reports were found.")
    return grouped


def load_e2e(path: Optional[str]) -> Dict[Tuple[str, str], float]:
    if not path:
        return {}

    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"model", "g", "e2e_s"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("E2E CSV must contain model,g,e2e_s columns.")

        for row in reader:
            value = number(row.get("e2e_s"))
            if value is None:
                continue
            key = (normalize_model(row.get("model")), normalize_g(row.get("g")))
            grouped[key].append(value)

    return {key: statistics.median(values) for key, values in grouped.items()}


def summarize(
    trials: Dict[Tuple[str, str], List[Dict[str, object]]],
    e2e: Dict[Tuple[str, str], float],
) -> List[Dict[str, object]]:
    medians: Dict[Tuple[str, str], Dict[str, object]] = {}
    for key, group in trials.items():
        labels = [str(item["label"]) for item in group]
        medians[key] = {
            "model": key[0],
            "model_label": statistics.mode(labels) if labels else key[0],
            "g": key[1],
            "trials": len(group),
            "jobs": median(item["jobs"] for item in group),
            "boundary_raw_s": median(item["boundary_s"] for item in group),
            "input_raw_mib": median(item["input_mib"] for item in group),
        }

    boundary_baselines = {
        model: float(row["boundary_raw_s"])
        for (model, g), row in medians.items()
        if g == "full" and row["boundary_raw_s"] is not None
    }
    input_baselines = {
        model: float(row["input_raw_mib"])
        for (model, g), row in medians.items()
        if g == "full" and row["input_raw_mib"] is not None
    }

    results: List[Dict[str, object]] = []
    for key in sorted(medians, key=lambda item: (item[0], g_sort_key(item[1]))):
        row = dict(medians[key])
        model, g = key
        boundary_baseline = boundary_baselines.get(model, 0.0)
        input_baseline = input_baselines.get(model, 0.0)

        overhead = float(row["boundary_raw_s"]) - boundary_baseline
        traffic = float(row["input_raw_mib"] or 0.0) - input_baseline
        if g == "full":
            overhead = 0.0
            traffic = 0.0

        measured_e2e = e2e.get(key)
        overhead_pct = (
            overhead / measured_e2e * 100.0 if measured_e2e else None
        )

        row.update(
            {
                "e2e_s": measured_e2e,
                "tensor_overhead_s": overhead,
                "tensor_overhead_pct_e2e": overhead_pct,
                "intermediate_input_mib": traffic,
            }
        )
        results.append(row)
    return results


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def markdown_table(results: List[Dict[str, object]]) -> str:
    lines = [
        "| Model | g | Jobs | Measured E2E (s) | Tensor overhead (s) | "
        "Tensor overhead / E2E | Intermediate input (MiB) | Repetitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in results:
        percentage = (
            f"{fmt(row['tensor_overhead_pct_e2e'], 1)}%"
            if row["tensor_overhead_pct_e2e"] is not None
            else "--"
        )
        lines.append(
            "| {model} | {g} | {jobs} | {e2e} | {overhead} | {pct} | "
            "{traffic} | {trials} |".format(
                model=row["model_label"],
                g=row["g"],
                jobs=fmt(row["jobs"], 0),
                e2e=fmt(row["e2e_s"]),
                overhead=fmt(row["tensor_overhead_s"]),
                pct=percentage,
                traffic=fmt(row["intermediate_input_mib"]),
                trials=row["trials"],
            )
        )
    return "\n".join(lines)


def conclusion(results: List[Dict[str, object]]) -> str:
    split_rows = [row for row in results if row["g"] != "full"]
    measured = [row for row in split_rows if row["e2e_s"] is not None]
    if not measured:
        return (
            "Measured E2E values were not supplied. The table quantifies tensor "
            "overhead, but an E2E crossover cannot be determined yet."
        )

    best = min(measured, key=lambda row: float(row["e2e_s"]))
    finest = min(measured, key=lambda row: float(row["g"]))
    if best["g"] == finest["g"]:
        crossover = (
            f"No crossover is observed within the tested range; g={best['g']} "
            "has the lowest measured E2E latency."
        )
    else:
        crossover = (
            f"The lowest measured E2E occurs at g={best['g']}; finer splitting "
            "beyond this point no longer compensates for its tensor overhead."
        )

    return (
        f"{crossover} At g={best['g']}, tensor handling contributes "
        f"{fmt(best['tensor_overhead_s'])} s "
        f"({fmt(best['tensor_overhead_pct_e2e'], 1)}% of E2E) and transfers "
        f"{fmt(best['intermediate_input_mib'])} MiB of intermediate inputs."
    )


CSV_FIELDS = [
    "model",
    "model_label",
    "g",
    "jobs",
    "e2e_s",
    "tensor_overhead_s",
    "tensor_overhead_pct_e2e",
    "intermediate_input_mib",
    "trials",
]


def write_csv(path: str, results: List[Dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    try:
        trials = load_transfer_results(args.inputs)
        e2e = load_e2e(args.e2e_csv)
        results = summarize(trials, e2e)

        table = markdown_table(results)
        result_text = conclusion(results)
        markdown = (
            "# Partition-boundary overhead\n\n"
            f"{table}\n\n"
            f"## Result\n\n{result_text}\n"
        )

        write_csv(args.output_csv, results)
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        sys.stdout.write(markdown)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

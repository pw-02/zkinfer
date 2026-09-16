#!/usr/bin/env python3
"""Benchmark EZKL verification for one monolithic proof and one split request.

The input is one JSON manifest. Paths may be absolute or relative to the
manifest file. Add ``swap_witness`` only when that proof must undergo EZKL
commitment swapping before verification.

Pass ``--parallel-workers N`` to additionally measure split-request wall time
with N persistent verifier processes. Process startup is excluded by warm-up.

Example manifest

{
  "full": [
    {
      "name": "full_model",
      "proof": "full/proof.pf",
      "settings": "full/settings.json",
      "vk": "full/vk.json",
      "srs": "srs/kzg17.srs"
    }
  ],
  "split": [
    {
      "name": "sub_model_1",
      "proof": "split/sub_model_1/proof.pf",
      "settings": "split/sub_model_1/settings.json",
      "vk": "split/sub_model_1/vk.json",
      "srs": "srs/kzg12.srs"
    },
    {
      "name": "sub_model_2",
      "proof": "split/sub_model_2/proof.pf",
      "swap_witness": "split/sub_model_2/chain_witness.json",
      "settings": "split/sub_model_2/settings.json",
      "vk": "split/sub_model_2/vk.json",
      "srs": "srs/kzg12.srs"
    }
  ]
}

For a chained job, ``swap_witness`` must contain the commitments that the
client is meant to substitute, including the upstream commitment used for the
downstream boundary. A job's ordinary witness is not automatically evidence
that two adjacent proofs are linked.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ezkl


@dataclass(frozen=True)
class ProofSpec:
    name: str
    proof: Path
    settings: Path
    vk: Path
    srs: Optional[Path]
    swap_witness: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Verification manifest JSON")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Measured repetitions after warm-up, default 10",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Unreported warm-up repetitions, default 1",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Also benchmark split proofs in parallel with this many processes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("verification_benchmark"),
        help="Directory for CSV and JSON results",
    )
    parser.add_argument(
        "--full-e2e-s",
        type=float,
        default=None,
        help="Optional measured full-model E2E latency in seconds",
    )
    parser.add_argument(
        "--split-e2e-s",
        type=float,
        default=None,
        help="Optional measured split-request E2E latency in seconds",
    )
    return parser.parse_args()


def resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_group(manifest_path: Path, data: Dict[str, Any], group: str) -> List[ProofSpec]:
    raw_specs = data.get(group)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Manifest field {group!r} must be a non-empty list")

    base = manifest_path.resolve().parent
    specs: List[ProofSpec] = []
    for index, raw in enumerate(raw_specs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{group}[{index}] must be a JSON object")

        missing = [key for key in ("proof", "settings", "vk") if not raw.get(key)]
        if missing:
            raise ValueError(f"{group}[{index}] is missing {', '.join(missing)}")

        specs.append(
            ProofSpec(
                name=str(raw.get("name") or f"{group}_{index}"),
                proof=resolve_path(base, raw["proof"]),
                settings=resolve_path(base, raw["settings"]),
                vk=resolve_path(base, raw["vk"]),
                srs=resolve_path(base, raw.get("srs")),
                swap_witness=resolve_path(base, raw.get("swap_witness")),
            )
        )

    return specs


def validate_specs(specs: Iterable[ProofSpec]) -> None:
    for spec in specs:
        paths = {
            "proof": spec.proof,
            "settings": spec.settings,
            "vk": spec.vk,
            "srs": spec.srs,
            "swap_witness": spec.swap_witness,
        }
        for label, path in paths.items():
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{spec.name} {label} not found at {path}")


def verify_one(spec: ProofSpec, scratch_dir: Path) -> Dict[str, Any]:
    # swap_proof_commitments may rewrite the proof, so never touch the original.
    local_proof = scratch_dir / f"{spec.name}.pf"
    shutil.copy2(spec.proof, local_proof)

    swap_time_s = 0.0
    if spec.swap_witness is not None:
        start = time.perf_counter()
        ezkl.swap_proof_commitments(
            proof_path=str(local_proof),
            witness_path=str(spec.swap_witness),
        )
        swap_time_s = time.perf_counter() - start

    verify_kwargs: Dict[str, Any] = {
        "proof_path": str(local_proof),
        "settings_path": str(spec.settings),
        "vk_path": str(spec.vk),
    }
    if spec.srs is not None:
        verify_kwargs["srs_path"] = str(spec.srs)

    start = time.perf_counter()
    verified = bool(ezkl.verify(**verify_kwargs))
    verify_time_s = time.perf_counter() - start

    if not verified:
        raise RuntimeError(f"EZKL verification failed for {spec.name}")

    return {
        "job": spec.name,
        "swapped": spec.swap_witness is not None,
        "swap_time_s": swap_time_s,
        "verify_time_s": verify_time_s,
        "total_time_s": swap_time_s + verify_time_s,
        "proof_bytes": spec.proof.stat().st_size,
        "vk_bytes": spec.vk.stat().st_size,
    }


def run_group(group: str, specs: List[ProofSpec], trial: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"zkinfer_verify_{group}_") as tmp:
        scratch_dir = Path(tmp)
        for spec in specs:
            rows.append(verify_one(spec, scratch_dir))

    return {
        "group": group,
        "mode": "sequential",
        "trial": trial,
        "jobs": len(specs),
        "parallel_workers": 1,
        "swaps": sum(int(row["swapped"]) for row in rows),
        "swap_time_s": sum(float(row["swap_time_s"]) for row in rows),
        "verify_time_s": sum(float(row["verify_time_s"]) for row in rows),
        "total_time_s": sum(float(row["total_time_s"]) for row in rows),
    }


def verify_one_isolated(spec: ProofSpec) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="zkinfer_verify_parallel_") as tmp:
        return verify_one(spec, Path(tmp))


def run_group_parallel(
    group: str,
    specs: List[ProofSpec],
    trial: int,
    executor: ProcessPoolExecutor,
    parallel_workers: int,
) -> Dict[str, Any]:
    start = time.perf_counter()
    rows = list(executor.map(verify_one_isolated, specs))
    wall_time_s = time.perf_counter() - start

    return {
        "group": group,
        "mode": "parallel",
        "trial": trial,
        "jobs": len(specs),
        "parallel_workers": parallel_workers,
        "swaps": sum(int(row["swapped"]) for row in rows),
        "swap_time_s": sum(float(row["swap_time_s"]) for row in rows),
        "verify_time_s": sum(float(row["verify_time_s"]) for row in rows),
        # For parallel execution, total_time_s is observed client wall time.
        "total_time_s": wall_time_s,
    }


def unique_bytes(specs: Iterable[ProofSpec], field: str) -> int:
    paths = {getattr(spec, field).resolve() for spec in specs if getattr(spec, field) is not None}
    return sum(path.stat().st_size for path in paths)


def median(rows: List[Dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def summarize(
    group: str,
    mode: str,
    specs: List[ProofSpec],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "group": group,
        "mode": mode,
        "jobs": len(specs),
        "parallel_workers": int(rows[0]["parallel_workers"]),
        "swap_calls": sum(spec.swap_witness is not None for spec in specs),
        "median_swap_time_s": median(rows, "swap_time_s"),
        "median_verify_time_s": median(rows, "verify_time_s"),
        "median_total_time_s": median(rows, "total_time_s"),
        "min_total_time_s": min(float(row["total_time_s"]) for row in rows),
        "max_total_time_s": max(float(row["total_time_s"]) for row in rows),
        "proof_bytes": unique_bytes(specs, "proof"),
        "verification_key_bytes": unique_bytes(specs, "vk"),
        "client_artifact_bytes": unique_bytes(specs, "proof") + unique_bytes(specs, "vk"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers cannot be negative")
    if args.full_e2e_s is not None and args.full_e2e_s <= 0:
        raise ValueError("--full-e2e-s must be positive")
    if args.split_e2e_s is not None and args.split_e2e_s <= 0:
        raise ValueError("--split-e2e-s must be positive")

    manifest_path = args.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    groups = {
        "full": load_group(manifest_path, manifest, "full"),
        "split": load_group(manifest_path, manifest, "split"),
    }
    validate_specs(spec for specs in groups.values() for spec in specs)

    trial_rows: List[Dict[str, Any]] = []
    executor = None
    try:
        if args.parallel_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=args.parallel_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )

        for _ in range(args.warmups):
            for group, specs in groups.items():
                run_group(group, specs, trial=0)
            if executor is not None:
                run_group_parallel(
                    "split",
                    groups["split"],
                    trial=0,
                    executor=executor,
                    parallel_workers=args.parallel_workers,
                )

        for trial in range(1, args.trials + 1):
            for group, specs in groups.items():
                row = run_group(group, specs, trial=trial)
                trial_rows.append(row)
                print(
                    f"trial={trial} group={group} mode=sequential jobs={row['jobs']} "
                    f"swaps={row['swaps']} swap={row['swap_time_s']:.6f}s "
                    f"verify={row['verify_time_s']:.6f}s total={row['total_time_s']:.6f}s"
                )

            if executor is not None:
                row = run_group_parallel(
                    "split",
                    groups["split"],
                    trial=trial,
                    executor=executor,
                    parallel_workers=args.parallel_workers,
                )
                trial_rows.append(row)
                print(
                    f"trial={trial} group=split mode=parallel jobs={row['jobs']} "
                    f"workers={row['parallel_workers']} wall={row['total_time_s']:.6f}s"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summaries = []
    for group, specs in groups.items():
        rows = [
            row
            for row in trial_rows
            if row["group"] == group and row["mode"] == "sequential"
        ]
        summaries.append(summarize(group, "sequential", specs, rows))
    if args.parallel_workers > 1:
        rows = [
            row
            for row in trial_rows
            if row["group"] == "split" and row["mode"] == "parallel"
        ]
        summaries.append(summarize("split", "parallel", groups["split"], rows))

    full_total = float(
        next(
            row
            for row in summaries
            if row["group"] == "full" and row["mode"] == "sequential"
        )["median_total_time_s"]
    )
    split_total = float(
        next(
            row
            for row in summaries
            if row["group"] == "split" and row["mode"] == "sequential"
        )["median_total_time_s"]
    )
    split_parallel_total = None
    if args.parallel_workers > 1:
        split_parallel_total = float(
            next(
                row
                for row in summaries
                if row["group"] == "split" and row["mode"] == "parallel"
            )["median_total_time_s"]
        )
    comparison = {
        "split_minus_full_time_s": split_total - full_total,
        "split_over_full_time_ratio": split_total / full_total if full_total else None,
        "split_parallel_time_s": split_parallel_total,
        "split_parallel_minus_full_time_s": (
            split_parallel_total - full_total
            if split_parallel_total is not None
            else None
        ),
        "split_parallel_over_full_time_ratio": (
            split_parallel_total / full_total
            if split_parallel_total is not None and full_total
            else None
        ),
        "full_verification_pct_of_e2e": (
            100.0 * full_total / args.full_e2e_s if args.full_e2e_s else None
        ),
        "split_verification_pct_of_e2e": (
            100.0 * split_total / args.split_e2e_s if args.split_e2e_s else None
        ),
        "split_parallel_verification_pct_of_e2e": (
            100.0 * split_parallel_total / args.split_e2e_s
            if args.split_e2e_s and split_parallel_total is not None
            else None
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "verification_trials.csv", trial_rows)
    write_csv(args.output_dir / "verification_summary.csv", summaries)
    with (args.output_dir / "verification_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "ezkl_version": getattr(ezkl, "__version__", "unknown"),
                "trials": args.trials,
                "warmups": args.warmups,
                "summary": summaries,
                "comparison": comparison,
            },
            handle,
            indent=2,
        )

    print("\nSummary")
    for row in summaries:
        print(
            f"{row['group']} {row['mode']}: jobs={row['jobs']} "
            f"workers={row['parallel_workers']} swap_calls={row['swap_calls']} "
            f"median={row['median_total_time_s']:.6f}s "
            f"proofs={row['proof_bytes'] / (1024 ** 2):.3f}MiB "
            f"proofs+VKs={row['client_artifact_bytes'] / (1024 ** 2):.3f}MiB"
        )
    print(
        f"split minus full={comparison['split_minus_full_time_s']:.6f}s "
        f"ratio={comparison['split_over_full_time_ratio']:.3f}x"
    )
    if comparison["split_parallel_time_s"] is not None:
        print(
            f"parallel split minus full={comparison['split_parallel_minus_full_time_s']:.6f}s "
            f"ratio={comparison['split_parallel_over_full_time_ratio']:.3f}x"
        )
    if comparison["split_verification_pct_of_e2e"] is not None:
        print(
            "split verification as E2E percentage="
            f"{comparison['split_verification_pct_of_e2e']:.6f}%"
        )
    if comparison["full_verification_pct_of_e2e"] is not None:
        print(
            "full verification as E2E percentage="
            f"{comparison['full_verification_pct_of_e2e']:.6f}%"
        )
    if comparison["split_parallel_verification_pct_of_e2e"] is not None:
        print(
            "parallel split verification as E2E percentage="
            f"{comparison['split_parallel_verification_pct_of_e2e']:.6f}%"
        )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

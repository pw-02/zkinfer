

#!/usr/bin/env python3
"""Benchmark EZKL verification for one monolithic proof and one split request.

The input is one JSON manifest. Paths may be absolute or relative to the
manifest file. Add ``swap_witness`` only when that proof must undergo EZKL
commitment swapping before verification.

Alternatively, pass ``--full-job-dir``, ``--split-jobs-dir``, and optionally
``--srs-dir`` to discover the standard zkInfer artifact layout directly.
Use ``--swap-split-witnesses`` when each split job's preserved witness contains
the commitments that should be substituted before verification.

If monolithic artifacts are not yet available, omit ``--full-job-dir`` and
pass a previously measured value with ``--full-verification-s``. If neither is
provided, the script runs the split benchmark without a monolithic comparison.

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
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ezkl
import ezkl
SPLIT_JOB_DIR = "e2e_runs/2026-09-16_15-18-48_nano_gpt_4_layers_64_embd_g1/requests/nano-gpt-4-layers-64-embd_split-fixed_ops-1_sched-lpt_simplified_2026-09-16_15-18-53/artifacts/jobs"
SRS_DIR = "srs"

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
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="Optional verification manifest JSON",
    )
    parser.add_argument(
        "--full-job-dir",
        type=Path,
        default=None,
        help="Monolithic job directory containing proof.pf, settings.json, and vk.json",
    )
    parser.add_argument(
        "--full-verification-s",
        type=float,
        default=None,
        help="Optional previously measured monolithic verification time",
    )
    parser.add_argument(
        "--split-jobs-dir",
        type=Path,
        default=SPLIT_JOB_DIR,
        help="Directory containing the split sub_model_* job directories",
    )
    parser.add_argument(
        "--srs-dir",
        type=Path,
        default=SRS_DIR,
        help="Directory containing kzg<logrows>.srs or ipa<logrows>.srs files",
    )
    parser.add_argument(
        "--download-missing-srs",
        default=True,
        action="store_true",
        help="Download required public SRS files that are absent from --srs-dir",
    )
    parser.add_argument(
        "--swap-split-witnesses",
        action="store_true",
        help="Run commitment swapping with each split job's witness.json",
    )
    parser.add_argument(
        "--swap-full-witness",
        action="store_true",
        help="Also run commitment swapping for the monolithic proof",
    )
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


def resolve_srs_from_settings(
    settings_path: Path,
    srs_dir: Optional[Path],
    download_missing_srs: bool,
) -> Optional[Path]:
    if srs_dir is None:
        if download_missing_srs:
            raise ValueError("--download-missing-srs requires --srs-dir")
        return None

    with settings_path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)

    run_args = settings.get("run_args") or {}
    logrows = run_args.get("logrows")
    if logrows is None:
        raise ValueError(f"{settings_path} does not contain run_args.logrows")

    commitment = str(run_args.get("commitment") or "kzg").lower()
    prefix = "ipa" if commitment == "ipa" else "kzg"
    resolved_srs_dir = Path(srs_dir).expanduser().resolve()
    resolved_srs_dir.mkdir(parents=True, exist_ok=True)
    srs_path = resolved_srs_dir / f"{prefix}{int(logrows)}.srs"
    if not srs_path.is_file() and download_missing_srs:
        print(f"Downloading public SRS to {srs_path}")
        downloaded = bool(
            ezkl.get_srs(
                settings_path=str(settings_path),
                srs_path=str(srs_path),
            )
        )
        if not downloaded:
            raise RuntimeError(f"SRS download failed for {settings_path}")
    if not srs_path.is_file():
        raise FileNotFoundError(
            f"SRS required by {settings_path} was not found at {srs_path}. "
            "Add --download-missing-srs to download it."
        )
    return srs_path


def proof_spec_from_job_dir(
    job_dir: Path,
    srs_dir: Optional[Path],
    swap_witness: bool,
    download_missing_srs: bool,
) -> ProofSpec:
    job_dir = job_dir.expanduser().resolve()
    settings_path = job_dir / "settings.json"
    witness_path = job_dir / "witness.json"
    return ProofSpec(
        name=job_dir.name,
        proof=job_dir / "proof.pf",
        settings=settings_path,
        vk=job_dir / "vk.json",
        srs=resolve_srs_from_settings(
            settings_path,
            srs_dir,
            download_missing_srs,
        ),
        swap_witness=witness_path if swap_witness else None,
    )


def submodel_sort_key(path: Path):
    match = re.search(r"sub_model_(\d+)", path.name)
    return (int(match.group(1)), path.name) if match else (10**12, path.name)


def discover_groups(args: argparse.Namespace) -> Dict[str, List[ProofSpec]]:
    if args.manifest is not None:
        if args.full_job_dir is not None or args.split_jobs_dir is not None:
            raise ValueError("Use either a manifest or the job-directory arguments, not both")
        manifest_path = args.manifest.resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return {
            "full": load_group(manifest_path, manifest, "full"),
            "split": load_group(manifest_path, manifest, "split"),
        }

    if args.split_jobs_dir is None:
        raise ValueError(
            "Provide a manifest or provide --split-jobs-dir"
        )

    split_root = args.split_jobs_dir.expanduser().resolve()
    if not split_root.is_dir():
        raise NotADirectoryError(f"Split jobs directory not found at {split_root}")

    split_dirs = sorted(
        (
            path
            for path in split_root.iterdir()
            if path.is_dir() and (path / "proof.pf").is_file()
        ),
        key=submodel_sort_key,
    )
    if not split_dirs:
        raise ValueError(f"No job directories containing proof.pf found in {split_root}")

    groups = {
        "split": [
            proof_spec_from_job_dir(
                job_dir,
                args.srs_dir,
                args.swap_split_witnesses,
                args.download_missing_srs,
            )
            for job_dir in split_dirs
        ],
    }
    if args.full_job_dir is not None:
        groups["full"] = [
            proof_spec_from_job_dir(
                args.full_job_dir,
                args.srs_dir,
                args.swap_full_witness,
                args.download_missing_srs,
            )
        ]
    elif args.swap_full_witness:
        raise ValueError("--swap-full-witness requires --full-job-dir")

    return groups


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
    if args.full_verification_s is not None and args.full_verification_s <= 0:
        raise ValueError("--full-verification-s must be positive")

    groups = discover_groups(args)
    if "full" in groups and args.full_verification_s is not None:
        raise ValueError(
            "Use either --full-job-dir or --full-verification-s, not both"
        )
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

    full_total = args.full_verification_s
    full_baseline_source = None
    if "full" in groups:
        full_total = float(
            next(
                row
                for row in summaries
                if row["group"] == "full" and row["mode"] == "sequential"
            )["median_total_time_s"]
        )
        full_baseline_source = "measured_in_this_run"
    elif full_total is not None:
        full_baseline_source = "provided_with_full_verification_s"
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
        "full_verification_time_s": full_total,
        "full_baseline_source": full_baseline_source,
        "split_minus_full_time_s": (
            split_total - full_total if full_total is not None else None
        ),
        "split_over_full_time_ratio": (
            split_total / full_total if full_total else None
        ),
        "split_parallel_time_s": split_parallel_total,
        "split_parallel_minus_full_time_s": (
            split_parallel_total - full_total
            if split_parallel_total is not None and full_total is not None
            else None
        ),
        "split_parallel_over_full_time_ratio": (
            split_parallel_total / full_total
            if split_parallel_total is not None and full_total
            else None
        ),
        "full_verification_pct_of_e2e": (
            100.0 * full_total / args.full_e2e_s
            if args.full_e2e_s and full_total is not None
            else None
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
    if full_total is not None:
        print(
            f"full baseline={full_total:.6f}s source={full_baseline_source}"
        )
        print(
            f"split minus full={comparison['split_minus_full_time_s']:.6f}s "
            f"ratio={comparison['split_over_full_time_ratio']:.3f}x"
        )
    else:
        print("No monolithic baseline supplied; split-only results were recorded.")
    if comparison["split_parallel_minus_full_time_s"] is not None:
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

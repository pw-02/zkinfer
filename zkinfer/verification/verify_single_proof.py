#!/usr/bin/env python3
"""Verify one EZKL proof stored in a job directory or ZIP file."""

import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

import ezkl


def verify(root: Path, srs_dir: Path) -> None:
    proof = next(root.rglob("proof.pf"))
    job = proof.parent
    settings = job / "settings.json"
    vk = job / "vk.json"

    settings_data = json.loads(settings.read_text(encoding="utf-8"))
    run_args = settings_data["run_args"]
    logrows = int(run_args["logrows"])
    commitment = str(run_args.get("commitment", "KZG")).lower()
    srs_name = f"{'ipa' if commitment == 'ipa' else 'kzg'}{logrows}.srs"
    srs = srs_dir if srs_dir.is_file() else srs_dir / srs_name

    for path in (proof, settings, vk, srs):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Verifying {job.name}")
    started = time.perf_counter()
    valid = bool(
        ezkl.verify(
            proof_path=str(proof),
            settings_path=str(settings),
            vk_path=str(vk),
            srs_path=str(srs),
        )
    )
    elapsed = time.perf_counter() - started

    print(f"Valid: {valid}")
    print(f"Verification time: {elapsed:.6f} seconds")
    if not valid:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", 
                        type=Path, help="Job directory or ZIP file",
                        default="e2e_runs/2026-09-16_15-18-48_nano_gpt_4_layers_64_embd_g1/requests/nano-gpt-4-layers-64-embd_split-fixed_ops-1_sched-lpt_simplified_2026-09-16_15-18-53/artifacts/jobs/sub_model_1_ad6462b4")
    parser.add_argument(
        "--srs_dir",
        type=Path,
        default="srs",
        help="SRS directory; the required filename is read from settings.json",
    )
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve()
    srs_dir = args.srs_dir.expanduser().resolve()

    if artifact.suffix.lower() != ".zip":
        verify(artifact, srs_dir)
        return

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(artifact) as archive:
            archive.extractall(tmp)
        verify(Path(tmp), srs_dir)


if __name__ == "__main__":
    main()

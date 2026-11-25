"""Immutable process configuration snapshots copied into individual runs."""
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

from omegaconf import OmegaConf


def code_revision():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    try:
        changes = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(changes),
                "tracked_changes": changes.splitlines()}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "tracked_changes": []}


def save_process_metadata(directory, role, runtime_config, resolved_config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    stem = f"{role}_{stamp.strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}"
    runtime = asdict(runtime_config)
    resolved = (OmegaConf.to_container(resolved_config, resolve=True)
                if resolved_config is not None else runtime)
    packages = {}
    for package in ("ezkl", "grpcio", "protobuf", "hydra-core"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata = {"role": role, "started_at": stamp.isoformat(), "pid": os.getpid(),
                "code": code_revision(), "packages": packages,
                "resolved_config": resolved, "runtime_config": runtime,
                "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS")}
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def copy_run_metadata(snapshot, directory):
    directory = Path(directory) / "config"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, directory / Path(snapshot).name)

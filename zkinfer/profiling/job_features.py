"""Checkpoint circuit metadata without requiring successful proving."""
import json
import os
from pathlib import Path

from zkinfer.profiling.msm_fft_parser import read_csv_first_row


def save_circuit_features(task, settings=None):
    root = Path(task.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "circuit_features.json"
    payload = json.loads(target.read_text()) if target.is_file() else {}
    payload.update(setup_key=task.setup_key, lease_id=task.lease_id,
                   profiling_mode=os.environ.get("ZKINFER_PROFILING_MODE", "research"),
                   rayon_num_threads=os.environ.get("RAYON_NUM_THREADS"))
    if settings is None:
        path = Path(getattr(task, "prepared_settings_path", None) or (Path(task.setup_dir) / "settings.json"))
        if path.is_file():
            settings = json.loads(path.read_text())
    if settings:
        payload["settings"] = settings
    early = root / "halo2_structure.json"
    if early.is_file():
        payload.setdefault("halo2", {}).update(json.loads(early.read_text()))
    payload.setdefault("halo2", {}).update(read_csv_first_row(str(root / "halo2_circuit.csv")))
    pool = root / "rayon_pool_size.txt"
    if pool.is_file():
        payload["rayon_pool_size"] = int(pool.read_text().strip())
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)
    return payload

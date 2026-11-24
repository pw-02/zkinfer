"""Render experiment YAML settings for the local tmux launcher."""
import shlex
from pathlib import Path

from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("experiments/config/config.yaml")
    node, storage = cfg.node, cfg.storage
    admission = node.stage_admission
    profiles = storage.resource_profiles
    values = {
        "WORKLOAD": cfg.suite.workloads[0],
        "HOST": cfg.launch.coordinator_host,
        "PORT": cfg.launch.coordinator_port,
        "ADMISSION": admission.enabled,
        "ADMISSION_MODE": admission.mode,
        "MAX_JOBS": node.max_concurrent_jobs,
        "KEEP_PROOFS": cfg.output.keep_proof_artifacts,
        "PROFILING_MODE": cfg.get("profiling", {}).get("mode", "research"),
        "SAMPLING": node.resource_sampling_interval_sec,
        "POLICY": cfg.coordinator.scheduling_policy,
        "RESERVE": admission.memory_safety_reserve_gb,
        "PERCENTILE": admission.history_percentile,
        "RUNS_DIR": Path(cfg.paths.runs_dir).resolve(),
        "REQUEST_DATA": Path(storage.request_data.path).resolve(),
        "SETUP": Path(storage.reusable_setup.path).resolve(),
        "CACHE": Path(storage.local_cache.path).resolve(),
        "CACHE_GB": storage.local_cache.max_gb,
        "REUSE": storage.reusable_setup.enabled,
        "PROFILES": profiles.enabled,
        "PROFILE_PATH": Path(profiles.path).resolve() if profiles.path else None,
    }
    if values["PROFILING_MODE"] not in ("basic", "research", "debug"):
        raise ValueError("Invalid profiling.mode")
    for name, value in values.items():
        value = "null" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
        print(f"CFG_{name}={shlex.quote(value)}")


if __name__ == "__main__":
    main()

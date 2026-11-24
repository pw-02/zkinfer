"""Translate the small experiment YAML into internal runtime settings."""
from pathlib import Path
from omegaconf import OmegaConf


def load_config(path="experiments/config/config.yaml", *, workload=None, run_dir=None):
    user = OmegaConf.load(path)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/runtime_experiment.yaml")
    selected = workload or user.workload
    if not isinstance(selected, str) or not selected or Path(selected).name != selected or selected in (".", ".."):
        raise ValueError("workload must be a workload YAML name")
    cfg.suite.workloads = [selected]
    cfg.execution.partitioning_enabled = user.partitioning.enabled
    cfg.execution.partition_size = user.partitioning.size
    cfg.execution.simplify_model = user.partitioning.get("simplify", True)
    if type(user.partitioning.size) is not int or user.partitioning.size < 1:
        raise ValueError("partitioning.size must be a positive integer")
    mode = user.scheduling.memory_admission
    if mode not in ("disabled", "whole_job", "stage"):
        raise ValueError("Invalid scheduling.memory_admission")
    cap = user.scheduling.max_concurrent_jobs
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("max_concurrent_jobs must be positive or null")
    if cap is None and mode == "disabled":
        raise ValueError("Unlimited concurrency requires memory admission")
    if user.scheduling.policy not in ("fifo", "lpt", "cache_aware", "resource_aware"):
        raise ValueError("Invalid scheduling.policy")
    cfg.coordinator.scheduling_policy = user.scheduling.policy
    cfg.node.max_concurrent_jobs = cap
    cfg.node.stage_admission.enabled = mode != "disabled"
    cfg.node.stage_admission.mode = "stage" if mode == "disabled" else mode
    cfg.node.stage_admission.memory_safety_reserve_gb = user.scheduling.memory_reserve_gib
    if user.profiling.mode not in ("basic", "research", "debug"):
        raise ValueError("Invalid profiling.mode")
    if user.profiling.sampling_interval_sec <= 0:
        raise ValueError("Sampling interval must be positive")
    cfg.profiling.mode = user.profiling.mode
    cfg.node.resource_sampling_interval_sec = user.profiling.sampling_interval_sec
    for value in (user.cache.setup, user.cache.profiles, user.output.keep_proofs, user.output.keep_cache):
        if type(value) is not bool:
            raise ValueError("Cache and output switches must be booleans")
    cfg.storage.reusable_setup.enabled = user.cache.setup
    cfg.storage.resource_profiles.enabled = user.cache.profiles
    cfg.storage.resource_profiles.path = "storage/profiles"
    cfg.output.keep_proof_artifacts = user.output.keep_proofs
    cfg.output.keep_request_data = user.output.keep_proofs
    cfg.output.keep_cache = user.output.keep_cache
    if run_dir is not None:
        root = Path(run_dir).resolve()
        cfg.paths.runs_dir = str(root)
        cfg.storage.request_data.path = str(root / "storage" / "requests")
        if not user.output.keep_cache:
            cache_root = root / "storage" / "cache"
            cfg.storage.reusable_setup.path = str(cache_root / "setup")
            cfg.storage.resource_profiles.path = str(cache_root / "profiles")
            cfg.storage.local_cache.path = str(cache_root / "node")
            cfg.node.runtime_state_dir = str(cache_root / "runtime")
        else:
            cfg.node.runtime_state_dir = "storage/node/runtime"
        # Do not carry node-local admission history into a no-profile run.
        if not user.cache.profiles:
            cfg.node.runtime_state_dir = str(root / "storage" / "runtime")
    return user, cfg

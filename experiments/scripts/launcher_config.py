"""Translate experiment-facing configuration into internal runtime settings."""
from __future__ import annotations

import math
import re
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = "experiments/config/config.yaml"
SCHEDULERS = ("fifo", "lpt", "cache_aware", "resource_aware")
ADMISSION_MODES = ("disabled", "whole_job", "stage")
PROFILING_MODES = ("basic", "research", "debug")


def _choice(value: str, field: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{field} must be one of: {', '.join(choices)}")


def _boolean(value: object, field: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")


def _positive_int(value: object, field: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _number(value: object, field: str, *, allow_zero: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        constraint = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be a finite {constraint} number")


def _validate(user: DictConfig) -> None:
    workload = user.workload
    if (
        not isinstance(workload, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", workload)
    ):
        raise ValueError("workload must be a workload YAML name without extension")
    if not (CONFIG_DIR / "workload" / f"{workload}.yaml").is_file():
        raise ValueError(f"Unknown workload: {workload}")

    _positive_int(user.partitioning.size, "partitioning.size")
    _choice(user.partitioning.get("witness_preparation", "upfront"),
            "partitioning.witness_preparation", ("independent", "upfront"))
    cap_scale = user.partitioning.get("max_propagated_scale", 0)
    if type(cap_scale) is not int or not 0 <= cap_scale <= 62:
        raise ValueError("partitioning.max_propagated_scale must be in [0, 62]")
    _boolean(user.partitioning.get("allow_boundary_rescale", True),
             "partitioning.allow_boundary_rescale")
    preparation = user.get("preparation", {})
    _choice(preparation.get("dispatch", "batch"), "preparation.dispatch", ("batch", "streaming"))
    if preparation.get("cache_enabled") is not None:
        _boolean(preparation["cache_enabled"], "preparation.cache_enabled")
    _choice(user.scheduling.policy, "scheduling.policy", SCHEDULERS)
    _choice(user.scheduling.memory_admission, "scheduling.memory_admission", ADMISSION_MODES)
    cap = user.scheduling.max_concurrent_jobs
    if cap is not None:
        _positive_int(cap, "scheduling.max_concurrent_jobs")
    elif user.scheduling.memory_admission == "disabled":
        raise ValueError("Unlimited concurrency requires memory admission")
    _number(user.scheduling.memory_reserve_gib, "scheduling.memory_reserve_gib", allow_zero=True)
    _choice(user.profiling.mode, "profiling.mode", PROFILING_MODES)
    _number(user.profiling.sampling_interval_sec, "profiling.sampling_interval_sec")

    for field in (
        "partitioning.enabled",
        "partitioning.simplify",
        "cache.setup",
        "cache.profiles",
        "output.keep_proofs",
        "output.keep_cache",
    ):
        _boolean(OmegaConf.select(user, field), field)


def _apply_execution(cfg: DictConfig, user: DictConfig) -> None:
    cfg.suite.workloads = [user.workload]
    cfg.execution.partitioning_enabled = user.partitioning.enabled
    cfg.execution.partition_size = user.partitioning.size
    cfg.execution.simplify_model = user.partitioning.simplify
    cfg.coordinator.witness_preparation = user.partitioning.get("witness_preparation", "upfront")
    cfg.coordinator.max_propagated_scale = user.partitioning.get("max_propagated_scale", 0)
    cfg.coordinator.allow_boundary_rescale = user.partitioning.get("allow_boundary_rescale", True)
    preparation = user.get("preparation", {})
    cfg.coordinator.preparation_dispatch = preparation.get("dispatch", "batch")
    cfg.coordinator.preparation_cache_enabled = preparation.get("cache_enabled", None)
    cfg.verification.enabled = cfg.coordinator.witness_preparation == "upfront"
    cfg.verification.trusted_vk_manifest = user.get("verification", {}).get("trusted_vk_manifest", None)


def _apply_scheduling(cfg: DictConfig, user: DictConfig) -> None:
    scheduling = user.scheduling
    admission = cfg.node.stage_admission
    mode = scheduling.memory_admission

    cfg.coordinator.scheduling_policy = scheduling.policy
    cfg.node.max_concurrent_jobs = scheduling.max_concurrent_jobs
    admission.enabled = mode != "disabled"
    admission.mode = "stage" if mode == "disabled" else mode
    admission.memory_safety_reserve_gb = scheduling.memory_reserve_gib


def _apply_storage(cfg: DictConfig, user: DictConfig) -> None:
    cfg.storage.reusable_setup.enabled = user.cache.setup
    cfg.storage.resource_profiles.enabled = user.cache.profiles
    cfg.storage.resource_profiles.path = "storage/profiles"
    cfg.output.keep_proof_artifacts = user.output.keep_proofs
    cfg.output.keep_request_data = user.output.keep_proofs
    cfg.output.keep_cache = user.output.keep_cache


def configure_run_paths(
    cfg: DictConfig,
    user: DictConfig,
    run_dir: str | Path,
) -> None:
    """Keep request data run-local and isolate disposable caches."""
    root = Path(run_dir).resolve()
    storage = root / "storage"

    cfg.paths.runs_dir = str(root)
    cfg.storage.request_data.path = str(storage / "requests")
    if user.output.keep_cache:
        cfg.node.runtime_state_dir = "storage/node/runtime"
    else:
        cache = storage / "cache"
        cfg.storage.reusable_setup.path = str(cache / "setup")
        cfg.storage.resource_profiles.path = str(cache / "profiles")
        cfg.storage.local_cache.path = str(cache / "node")
        cfg.node.runtime_state_dir = str(cache / "runtime")

    # Do not carry node-local admission history into a no-profile run.
    if not user.cache.profiles:
        cfg.node.runtime_state_dir = str(storage / "runtime")


def make_run_name(
    user: DictConfig,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    """Describe requested settings, not observed cache hits or run outcomes."""
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    partition = f"p{user.partitioning.size}" if user.partitioning.enabled else "mono"
    setup = "setup-on" if user.cache.setup else "setup-off"
    cap = user.scheduling.max_concurrent_jobs
    concurrency = f"c{cap}" if cap is not None else "cauto"
    admission = {
        "disabled": "noadm",
        "whole_job": "whole",
        "stage": "stage",
    }[user.scheduling.memory_admission]
    policy = {
        "fifo": "fifo",
        "lpt": "lpt",
        "cache_aware": "cache",
        "resource_aware": "resource",
    }[user.scheduling.policy]
    unique = suffix if suffix is not None else uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", unique):
        raise ValueError("Invalid run suffix")
    return "_".join((
        timestamp, user.workload, partition, setup, concurrency,
        policy, admission,
        *(() if user.partitioning.get("witness_preparation", "upfront") == "independent" else ("exact",)),
        *(() if "preparation" not in user else (
            user.preparation.get("dispatch", "batch"),
            "prep-on" if user.preparation.get("cache_enabled", user.cache.setup) else "prep-off",
        )),
        unique,
    ))


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    workload: str | None = None,
    run_dir: str | Path | None = None,
) -> tuple[DictConfig, DictConfig]:
    """Load, validate and translate the effective experiment configuration."""
    user = OmegaConf.load(path)
    if workload is not None:
        user.workload = workload
    if "simplify" not in user.partitioning:
        user.partitioning.simplify = True
    _validate(user)

    cfg = OmegaConf.load(CONFIG_DIR / "runtime_experiment.yaml")
    _apply_execution(cfg, user)
    _apply_scheduling(cfg, user)
    _apply_storage(cfg, user)
    cfg.profiling.mode = user.profiling.mode
    cfg.node.resource_sampling_interval_sec = user.profiling.sampling_interval_sec

    if run_dir is not None:
        configure_run_paths(cfg, user, run_dir)
    return user, cfg

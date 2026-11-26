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
SCHEDULERS = ("fifo", "lpt", "resource_aware")
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


def _reuse(user: DictConfig, kind: str) -> bool:
    return user.cache[kind]


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
    if "witness_preparation" in user.partitioning:
        raise ValueError("Remove partitioning.witness_preparation; linked preparation is mandatory")
    cap_scale = user.partitioning.get("max_propagated_scale", 0)
    if type(cap_scale) is not int or not 0 <= cap_scale <= 62:
        raise ValueError("partitioning.max_propagated_scale must be in [0, 62]")
    _boolean(user.partitioning.get("allow_boundary_rescale", True),
             "partitioning.allow_boundary_rescale")
    preparation = user.get("preparation", {})
    _choice(preparation.get("dispatch", "batch"), "preparation.dispatch", ("batch", "streaming"))
    _positive_int(preparation.get("workers", 1), "preparation.workers")
    pending = preparation.get("max_pending", 8)
    if type(pending) is not int or pending < 0:
        raise ValueError("preparation.max_pending must be a non-negative integer")
    for kind in ("coordinator_threads", "prover_threads"):
        value = user.get("resources", {}).get(kind)
        if value is not None:
            _positive_int(value, f"resources.{kind}")
    _choice(user.get("proving", {}).get("key_representation", "full"),
            "proving.key_representation", ("full", "reduced", "compact"))
    for kind in ("circuits", "keys"):
        _boolean(_reuse(user, kind), f"cache.{kind}")
    for kind in ("use_history", "update_history"):
        _boolean(user.resource_model[kind], f"resource_model.{kind}")
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
        "output.keep_proofs",
    ):
        _boolean(OmegaConf.select(user, field), field)


def configure_run_paths(cfg: DictConfig, user: DictConfig, run_dir: str | Path) -> None:
    """Keep request artifacts and coordinator state private to each run."""
    root = Path(run_dir).resolve()
    cfg.paths.runs_dir = str(root)
    cfg.storage.request_path = str(root / ".runtime" / "storage" / "requests")
    cfg.coordinator.runs_dir = str(root / "requests")
    cfg.node.runs_dir = str(root / "requests")
    cfg.coordinator.logs_dir = str(root / "logs")
    cfg.node.logs_dir = str(root / "logs")
    cfg.coordinator.state_db_path = str(root / ".runtime" / "coordinator_state.sqlite3")


def make_run_name(
    user: DictConfig,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    """Describe requested settings, not observed cache hits or run outcomes."""
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    partition = f"p{user.partitioning.size}" if user.partitioning.enabled else "mono"
    cap = user.scheduling.max_concurrent_jobs
    concurrency = f"c{cap}" if cap is not None else "cauto"
    policy = {"fifo": "fifo", "lpt": "lpt", "resource_aware": "resource"}[user.scheduling.policy]
    unique = suffix if suffix is not None else uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", unique):
        raise ValueError("Invalid run suffix")
    return "_".join((timestamp, user.workload, partition, concurrency, policy, unique))


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

    defaults = OmegaConf.load(CONFIG_DIR.parents[1] / "zkinfer/config/config.yaml")
    cfg = OmegaConf.merge(defaults, user, {
        "suite": {"workloads": [user.workload]},
        "node": {"node_id": "local-node"},
        "paths": {"runs_dir": "experiments/runs"},
        "launch": {"coordinator_ready_timeout_sec": 30, "node_startup_delay_sec": 1,
                   "poll_completion_interval_sec": 5},
    })

    if run_dir is not None:
        configure_run_paths(cfg, user, run_dir)
    return user, cfg

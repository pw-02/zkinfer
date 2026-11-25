from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


@dataclass(frozen=True)
class CoordinatorConfig:
    host: str
    port: int
    grpc_thread_pool_size: int
    grpc_max_message_mb: int
    runs_dir: str
    logs_dir: str
    witness_preparation: str = "independent"
    max_propagated_scale: int = 0
    allow_boundary_rescale: bool = True
    preparation_dispatch: str = "batch"
    preparation_cache_enabled: Optional[bool] = None


@dataclass(frozen=True)
class StageAdmissionConfig:
    """Node-local memory admission policy for proof execution stages."""

    enabled: bool
    memory_safety_reserve_gb: float
    history_percentile: float
    mode: str = "stage"


@dataclass(frozen=True)
class NodeConfig:
    node_id: Optional[str]
    runs_dir: str
    logs_dir: str
    job_poll_interval_sec: int
    resource_sampling_interval_sec: float
    max_concurrent_jobs: Optional[int]
    stage_admission: StageAdmissionConfig
    local_cache_dir: str
    local_cache_max_bytes: int
    runtime_state_dir: str


@dataclass(frozen=True)
class RequestStoreConfig:
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str


@dataclass(frozen=True)
class SetupStoreConfig:
    """Reusable partition model and derived proving state."""

    enabled: bool
    backend: str
    root_dir: str
    s3_bucket: Optional[str]
    s3_prefix: str
    resource_profiles_enabled: bool = True
    resource_profiles_root_dir: Optional[str] = None
    resource_profiles_update_history: Optional[bool] = None


@dataclass(frozen=True)
class JobConfig:
    scheduling_policy: str
    max_retries: int


@dataclass(frozen=True)
class HeartbeatConfig:
    interval_sec: int
    timeout_sec: int


@dataclass(frozen=True)
class StateStoreConfig:
    enabled: bool
    db_path: str


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool
    srs_dir: str
    require_split_chain: bool
    write_request_manifest: bool
    trusted_vk_manifest: Optional[str] = None


@dataclass(frozen=True)
class RuntimeConfig:
    coordinator: CoordinatorConfig
    node: NodeConfig
    request_store: RequestStoreConfig
    setup_store: SetupStoreConfig
    jobs: JobConfig
    heartbeat: HeartbeatConfig
    state_store: StateStoreConfig
    verification: VerificationConfig


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


def _gb_to_bytes(value) -> int:
    return int(float(value or 0) * (1024**3))


def _resolve_request_store_config(cfg: DictConfig) -> RequestStoreConfig:
    backend = str(cfg.storage.backend)
    request_path = str(cfg.storage.request_data.get("path", "storage/requests"))

    if backend == "filesystem":
        return RequestStoreConfig(
            backend="filesystem",
            root_dir=str(_resolve_path(request_path)),
            s3_bucket=None,
            s3_prefix="",
        )
    if backend == "s3":
        base_prefix = str(cfg.storage.s3_prefix).strip("/")
        object_prefix = request_path.strip("/")
        full_prefix = "/".join(part for part in (base_prefix, object_prefix) if part)
        return RequestStoreConfig(
            backend="s3",
            root_dir=full_prefix,
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=full_prefix,
        )
    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_setup_store_config(cfg: DictConfig) -> SetupStoreConfig:
    backend = str(cfg.storage.backend)
    reusable_setup = cfg.storage.reusable_setup
    enabled = bool(reusable_setup.get("enabled", False))
    setup_path = str(reusable_setup.get("path", "storage/setup"))
    profiles = cfg.storage.get("resource_profiles", {})
    profiles_enabled = bool(profiles.get("enabled", True))
    profiles_update = bool(profiles.get("update_history", profiles_enabled))
    profiles_path = str(profiles.get("path") or setup_path)

    if backend == "filesystem":
        return SetupStoreConfig(
            enabled=enabled,
            backend="filesystem",
            root_dir=str(_resolve_path(setup_path)),
            resource_profiles_enabled=profiles_enabled,
            resource_profiles_update_history=profiles_update,
            resource_profiles_root_dir=str(_resolve_path(profiles_path)),
            s3_bucket=None,
            s3_prefix="",
        )
    if backend == "s3":
        base_prefix = str(cfg.storage.s3_prefix).strip("/")
        relative_prefix = setup_path.strip("/")
        full_prefix = "/".join(part for part in (base_prefix, relative_prefix) if part)
        return SetupStoreConfig(
            enabled=enabled,
            backend="s3",
            root_dir=full_prefix,
            resource_profiles_enabled=profiles_enabled,
            resource_profiles_update_history=profiles_update,
            resource_profiles_root_dir="/".join(
                part for part in (base_prefix, profiles_path.strip("/")) if part
            ),
            s3_bucket=cfg.storage.s3_bucket,
            s3_prefix=full_prefix,
        )
    raise ValueError(f"Unsupported storage.backend: {backend}")


def _resolve_state_store_config(cfg: DictConfig) -> StateStoreConfig:
    db_path = _resolve_path(
        cfg.coordinator.get(
            "state_db_path",
            "experiments/runs/coordinator_state.sqlite3",
        )
    )
    return StateStoreConfig(
        enabled=bool(cfg.coordinator.get("state_store_enabled", True)),
        db_path=str(db_path),
    )


def _resolve_verification_config(cfg: DictConfig) -> VerificationConfig:
    verification = cfg.get("verification", {})
    return VerificationConfig(
        enabled=bool(verification.get("enabled", cfg.coordinator.get("witness_preparation", "upfront") == "upfront")),
        srs_dir=str(_resolve_path(verification.get("srs_dir", "srs"))),
        require_split_chain=bool(verification.get("require_split_chain", False)),
        write_request_manifest=bool(verification.get("write_request_manifest", True)),
        trusted_vk_manifest=(
            str(_resolve_path(verification.trusted_vk_manifest))
            if verification.get("trusted_vk_manifest") else None
        ),
    )


def _resolve_stage_admission_config(cfg: DictConfig) -> StageAdmissionConfig:
    admission = cfg.node.get("stage_admission", {})
    mode = str(admission.get("mode", "stage"))
    if mode not in {"stage", "whole_job"}:
        raise ValueError("node.stage_admission.mode must be stage or whole_job")
    return StageAdmissionConfig(
        mode=mode,
        enabled=bool(admission.get("enabled", True)),
        memory_safety_reserve_gb=max(
            0.0,
            float(admission.get("memory_safety_reserve_gb", 8.0)),
        ),
        history_percentile=min(
            1.0,
            max(0.5, float(admission.get("history_percentile", 0.90))),
        ),
    )


def _resolve_max_concurrent_jobs(cfg: DictConfig) -> Optional[int]:
    value = cfg.node.get("max_concurrent_jobs", None)
    if value is None:
        return None
    return max(1, int(value))


def build_runtime_config(cfg: DictConfig, process_role: Optional[str] = None) -> RuntimeConfig:
    if cfg.storage.backend not in {"filesystem", "s3"}:
        raise ValueError("storage.backend must be either 'filesystem' or 's3'")
    if cfg.storage.backend == "s3" and not cfg.storage.s3_bucket:
        raise ValueError("storage.s3_bucket must be set when storage.backend=s3")
    if process_role not in {None, "coordinator", "node"}:
        raise ValueError("process_role must be one of: coordinator, node, None")

    coordinator_runs_dir = _resolve_path(cfg.coordinator.runs_dir)
    coordinator_logs_dir = _resolve_path(cfg.coordinator.logs_dir)
    node_runs_dir = _resolve_path(cfg.node.runs_dir)
    node_logs_dir = _resolve_path(cfg.node.logs_dir)

    node_cache_cfg = cfg.node.get("local_cache", {})
    node_cache_dir = _resolve_path(
        node_cache_cfg.get("path", "storage/node/cache")
    )
    node_runtime_state_dir = _resolve_path(
        cfg.node.get("runtime_state_dir", "storage/node/runtime")
    )

    request_store = _resolve_request_store_config(cfg)
    setup_store = _resolve_setup_store_config(cfg)
    profiles_root = setup_store.resource_profiles_root_dir
    if (setup_store.resource_profiles_enabled or setup_store.resource_profiles_update_history) and profiles_root:
        if request_store.backend == "filesystem":
            request_path = Path(request_store.root_dir).resolve()
            profile_path = Path(profiles_root).resolve()
            overlaps = (request_path == profile_path or request_path in profile_path.parents
                        or profile_path in request_path.parents)
        else:
            request_path = request_store.root_dir.rstrip("/")
            profile_path = profiles_root.rstrip("/")
            overlaps = (request_path == profile_path or request_path.startswith(profile_path + "/")
                        or profile_path.startswith(request_path + "/"))
        if overlaps:
            raise ValueError("Resource profiles and request-data roots must be separate")
    if request_store.backend == "filesystem":
        request_root = Path(request_store.root_dir).resolve()
        setup_root = Path(setup_store.root_dir).resolve()
        if (
            request_root == setup_root
            or request_root in setup_root.parents
            or setup_root in request_root.parents
        ):
            raise ValueError("Reusable setup and request-data roots must be separate")
    else:
        request_root = request_store.root_dir.rstrip("/")
        setup_root = setup_store.root_dir.rstrip("/")
        if (
            request_root == setup_root
            or request_root.startswith(setup_root + "/")
            or setup_root.startswith(request_root + "/")
        ):
            raise ValueError("Reusable setup and request-data prefixes must be separate")

    state_store = _resolve_state_store_config(cfg)
    verification = _resolve_verification_config(cfg)
    stage_admission = _resolve_stage_admission_config(cfg)

    if process_role == "coordinator":
        coordinator_runs_dir.mkdir(parents=True, exist_ok=True)
        coordinator_logs_dir.mkdir(parents=True, exist_ok=True)
        Path(state_store.db_path).parent.mkdir(parents=True, exist_ok=True)
        if request_store.backend == "filesystem":
            Path(request_store.root_dir).mkdir(parents=True, exist_ok=True)
        if setup_store.backend == "filesystem":
            Path(setup_store.root_dir).mkdir(parents=True, exist_ok=True)
    elif process_role == "node":
        node_runs_dir.mkdir(parents=True, exist_ok=True)
        node_logs_dir.mkdir(parents=True, exist_ok=True)
        node_runtime_state_dir.mkdir(parents=True, exist_ok=True)

    preparation_dispatch = str(cfg.coordinator.get("preparation_dispatch", "batch"))
    if preparation_dispatch not in {"batch", "streaming"}:
        raise ValueError("coordinator.preparation_dispatch must be batch or streaming")
    preparation_cache_enabled = cfg.coordinator.get("preparation_cache_enabled", None)
    if preparation_cache_enabled is not None and type(preparation_cache_enabled) is not bool:
        raise ValueError("coordinator.preparation_cache_enabled must be boolean or null")
    if preparation_dispatch == "streaming" and int(cfg.coordinator.grpc_thread_pool_size) < 2:
        raise ValueError("Streaming preparation requires at least two gRPC threads")
    witness_preparation = str(cfg.coordinator.get("witness_preparation", "upfront"))
    if witness_preparation not in {"independent", "upfront"}:
        raise ValueError("coordinator.witness_preparation must be independent or upfront")
    if witness_preparation == "upfront":
        if not verification.enabled:
            raise ValueError("Upfront witness preparation requires verification.enabled=true")
        if request_store.backend != "filesystem" or setup_store.backend != "filesystem":
            raise ValueError("Upfront witness preparation currently requires filesystem storage")
    scale_cap = cfg.coordinator.get("max_propagated_scale", 0)
    if type(scale_cap) is not int or not 0 <= scale_cap <= 62:
        raise ValueError("max_propagated_scale must be an integer in [0, 62]")

    return RuntimeConfig(
        coordinator=CoordinatorConfig(
            host=cfg.coordinator.host,
            port=int(cfg.coordinator.port),
            grpc_thread_pool_size=int(cfg.coordinator.grpc_thread_pool_size),
            grpc_max_message_mb=int(cfg.coordinator.grpc_max_message_mb),
            runs_dir=str(coordinator_runs_dir),
            logs_dir=str(coordinator_logs_dir),
            witness_preparation=witness_preparation,
            max_propagated_scale=scale_cap,
            allow_boundary_rescale=bool(cfg.coordinator.get("allow_boundary_rescale", True)),
            preparation_dispatch=preparation_dispatch,
            preparation_cache_enabled=preparation_cache_enabled,
        ),
        node=NodeConfig(
            node_id=cfg.node.node_id,
            runs_dir=str(node_runs_dir),
            logs_dir=str(node_logs_dir),
            job_poll_interval_sec=int(cfg.node.job_poll_interval_sec),
            resource_sampling_interval_sec=max(
                0.05,
                float(cfg.node.resource_sampling_interval_sec),
            ),
            max_concurrent_jobs=_resolve_max_concurrent_jobs(cfg),
            stage_admission=stage_admission,
            local_cache_dir=str(node_cache_dir),
            local_cache_max_bytes=_gb_to_bytes(node_cache_cfg.get("max_gb", 100)),
            runtime_state_dir=str(node_runtime_state_dir),
        ),
        request_store=request_store,
        setup_store=setup_store,
        jobs=JobConfig(
            scheduling_policy=str(cfg.coordinator.scheduling_policy),
            max_retries=int(cfg.jobs.max_retries),
        ),
        heartbeat=HeartbeatConfig(
            interval_sec=int(cfg.heartbeat.interval_sec),
            timeout_sec=int(cfg.heartbeat.timeout_sec),
        ),
        state_store=state_store,
        verification=verification,
    )

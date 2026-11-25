import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence

from zkinfer.runtime.data_plane import ArtifactRef


class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RequestStatus(str, Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class NodeResources:
    """Fresh node state reported while asking the coordinator for work."""

    node_id: str
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    cpu_total_cores: float = 0.0
    cpu_available_cores: float = 0.0
    cache_limit_gb: float = 0.0
    cache_used_gb: float = 0.0
    active_job_ids: Sequence[str] = field(default_factory=tuple)
    cached_setup_keys: Sequence[str] = field(default_factory=tuple)

    @property
    def active_job_count(self) -> int:
        return len(self.active_job_ids or [])

    @property
    def cache_available_gb(self) -> float:
        if self.cache_limit_gb <= 0:
            return float("inf")
        return max(0.0, self.cache_limit_gb - self.cache_used_gb)


@dataclass
class ProofJob:
    """A single backend proving task produced from one inference request."""

    inference_request_name: str
    model_name: str
    inference_request_id: str
    model_ref: ArtifactRef
    input_ref: ArtifactRef
    setup_cache_ref: ArtifactRef
    output_prefix_ref: ArtifactRef
    profiling_file_path: Optional[str] = None
    model_write_time: float = 0.0
    input_write_time: float = 0.0
    profiling_data: Dict = field(default_factory=dict)
    predicted_duration_s: float = 0.0
    predicted_peak_memory_gb: float = 0.0
    artifact_key: str = ""
    max_retries: int = 0
    job_id: str = field(init=False)
    job_name: str = field(init=False)
    job_status: JobStatus = JobStatus.PREPARING
    queued_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    artifact_manifest_ref: Optional[ArtifactRef] = None
    assigned_node_id: Optional[str] = None
    lease_id: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    parent_model_hash: Optional[str] = None
    model_hash: Optional[str] = None

    @property
    def predicted_duration(self) -> float:
        return self.predicted_duration_s

    @property
    def model_path(self) -> str:
        return self.model_ref.uri

    @property
    def input_path(self) -> str:
        return self.input_ref.uri

    @property
    def cache_path(self) -> str:
        return self.setup_cache_ref.uri

    def __post_init__(self) -> None:
        self.job_id = f"{self.model_name}_{uuid.uuid4().hex[:8]}"
        self.job_name = f"{self.inference_request_name}_{self.model_name}"
        if not self.artifact_key:
            self.artifact_key = self.model_hash or self.setup_cache_ref.uri or self.model_name


@dataclass
class InferenceRequest:
    name: str
    onnx_model_path: str
    input_data_path: str
    split_mode: str
    ops_per_chunk: int
    # Snapshot of the coordinator-wide policy used for this request. Clients do
    # not choose this value independently.
    scheduler: str
    simplify_model: bool = False
    input_shapes: Optional[Dict] = None
    request_id: str = field(init=False)
    proof_jobs: List[ProofJob] = field(default_factory=list)
    preparation_metrics: Dict = field(default_factory=dict)
    # Streaming requests are open until every planned job has been published.
    preparation_complete: bool = True
    expected_job_count: Optional[int] = None
    preparation_error: Optional[str] = None
    created_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    queued_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    request_status: RequestStatus = RequestStatus.CREATED
    run_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    reports_dir: Optional[str] = None

    def __post_init__(self) -> None:
        # Experiment/campaign directories already record workload, partitioning,
        # scheduler and timestamps. Keep the request directory identifier short
        # so nested run paths remain readable.
        self.request_id = f"req_{uuid.uuid4().hex[:12]}"

    def all_jobs_finished(self) -> bool:
        return (self.preparation_complete
                and (self.preparation_error is not None or self.expected_job_count is None
                     or len(self.proof_jobs) == self.expected_job_count)
                and all(job.job_status in {JobStatus.COMPLETED, JobStatus.FAILED} for job in self.proof_jobs))

    def any_job_failed(self) -> bool:
        return any(job.job_status == JobStatus.FAILED for job in self.proof_jobs)

    def compute_progress(self) -> float:
        """Completed-job fraction retained for compatibility/reporting."""
        if not self.proof_jobs:
            return 0.0
        completed_jobs = sum(
            1 for job in self.proof_jobs if job.job_status == JobStatus.COMPLETED
        )
        return completed_jobs / len(self.proof_jobs) * 100.0

    def predicted_total_work_s(self) -> float:
        return sum(max(0.0, float(job.predicted_duration_s or 0.0)) for job in self.proof_jobs)

    def predicted_completed_work_s(self) -> float:
        return sum(
            max(0.0, float(job.predicted_duration_s or 0.0))
            for job in self.proof_jobs
            if job.job_status == JobStatus.COMPLETED
        )

    def predicted_remaining_work_s(self) -> float:
        return sum(
            max(0.0, float(job.predicted_duration_s or 0.0))
            for job in self.proof_jobs
            if job.job_status in {JobStatus.PREPARING, JobStatus.QUEUED, JobStatus.IN_PROGRESS}
        )

    def predicted_remaining_bottleneck_s(self) -> float:
        pending = [
            max(0.0, float(job.predicted_duration_s or 0.0))
            for job in self.proof_jobs
            if job.job_status in {JobStatus.PREPARING, JobStatus.QUEUED, JobStatus.IN_PROGRESS}
        ]
        return max(pending, default=0.0)

    def compute_work_progress(self) -> float:
        """Progress weighted by predicted proof work rather than job count.

        Decomposed requests can contain many cheap jobs and a few dominant jobs,
        so a completed-job percentage can substantially overstate real progress.
        When no duration history is available, fall back to the job-count metric.
        """
        total = self.predicted_total_work_s()
        if total <= 0.0:
            return self.compute_progress()
        return 100.0 * self.predicted_completed_work_s() / total

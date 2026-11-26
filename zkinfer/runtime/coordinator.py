from __future__ import annotations
from zkinfer.profiling.run_metadata import save_process_metadata

import logging
import copy
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from zkinfer.config.runtime import RuntimeConfig
from zkinfer.profiling.overhead import write_transfer_overhead_report
from zkinfer.profiling.reports import write_job_report, write_request_report, write_failed_attempt_report
from zkinfer.runtime.models import (
    ArtifactRef,
    InferenceRequest,
    JobStatus,
    NodeResources,
    ProofJob,
    RequestStatus,
)
from zkinfer.runtime.request_builder import RequestBuilder
from zkinfer.runtime.preparation_runner import run_preparation
from zkinfer.runtime.resource_profiles import get_profile_estimates, max_stage_memory_gb
from zkinfer.runtime.scheduler import JobScheduler
from zkinfer.runtime.state_store import CoordinatorStateStore
from zkinfer.storage.shared_store import SharedStorage
from zkinfer.storage.io import save_json
from zkinfer.verification.request_manifest import write_request_artifact_manifest
from zkinfer.verification.verifier import RuntimeVerifier
from zkinfer.verification.verification_runner import run_verification


class PreparationQueueFull(RuntimeError):
    """The coordinator cannot accept more preparation work right now."""


class Coordinator:
    """Global control plane for distributed proof jobs."""

    def __init__(self, runtime_config: RuntimeConfig, logger: logging.Logger, resolved_config=None):
        self.logger = logger
        self.metadata_snapshot = save_process_metadata(
            Path(runtime_config.coordinator.runs_dir).parent / ".runtime" / "processes",
            "coordinator", runtime_config, resolved_config)
        self.runs_dir = Path(runtime_config.coordinator.runs_dir)
        self.request_store_config = runtime_config.request_store
        self.setup_store = runtime_config.setup_store
        self.jobs_config = runtime_config.jobs
        self.heartbeat_config = runtime_config.heartbeat
        self.verification_config = runtime_config.verification
        self.scheduler_policy = self.jobs_config.scheduling_policy or "fifo"
        self.max_retries = self.jobs_config.max_retries

        self.shared_storage = SharedStorage(
            backend=self.request_store_config.backend,
            setup_root=self.setup_store.root_dir,
            requests_root=self.request_store_config.root_dir,
            s3_bucket=self.request_store_config.s3_bucket,
        )
        self.state_store = (
            CoordinatorStateStore(runtime_config.state_store.db_path)
            if runtime_config.state_store.enabled
            else None
        )
        self.verifier = (
            RuntimeVerifier(
                srs_dir=self.verification_config.srs_dir,
                enabled=self.verification_config.enabled,
                require_split_chain=self.verification_config.require_split_chain,
                write_request_manifest=False,
                trusted_vk_manifest=getattr(self.verification_config, "trusted_vk_manifest", None),
            )
            if self.verification_config.enabled
            else None
        )

        self.active_requests: Dict[str, InferenceRequest] = {}
        self.active_jobs: Dict[str, ProofJob] = {}
        self.node_heartbeats: Dict[str, datetime] = {}
        self.node_status: Dict[str, dict] = {}
        self.scheduler = JobScheduler(policy=self.scheduler_policy)
        self.scheduler_policy = self.scheduler.policy
        self.request_builder = RequestBuilder(
            logger=logger,
            witness_preparation=getattr(runtime_config.coordinator, "witness_preparation", "independent"),
            max_propagated_scale=getattr(runtime_config.coordinator, "max_propagated_scale", 16),
            allow_boundary_rescale=getattr(runtime_config.coordinator, "allow_boundary_rescale", True),
            preparation_cache_enabled=getattr(runtime_config.coordinator, "preparation_cache_enabled", None),
            prover_threads=runtime_config.node.rayon_threads,
        )
        self.preparation_dispatch = getattr(runtime_config.coordinator, "preparation_dispatch", "batch")
        self.preparation_rayon_threads = runtime_config.coordinator.rayon_threads
        if self.request_builder.witness_preparation == "upfront" and not self.verifier:
            raise ValueError("Upfront witness preparation requires an enabled verifier")
        self.lock = threading.RLock()
        self.profile_locks: Dict[str, threading.Lock] = {}
        self.pending_job_reports: set[str] = set()

        # Threads supervise fresh preparation subprocesses; no native preparation
        # runs inside the coordinator. The semaphore
        # bounds running plus queued work; ThreadPoolExecutor alone is unbounded.
        workers = runtime_config.coordinator.preparation_workers
        pending = runtime_config.coordinator.preparation_max_pending
        self.preparation_slots = threading.BoundedSemaphore(workers + pending)
        self.preparation_pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="request-preparer",
        )
        self.accepting_requests = True

        self.finalization_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="request-finalizer",
        )
        self.finalizing_requests: set[str] = set()

        self._recover_state()

    def shutdown(self) -> None:
        with self.lock:
            self.accepting_requests = False
        # Preparation may schedule finalization; drain it first.
        self.preparation_pool.shutdown(wait=True)
        self.finalization_pool.shutdown(wait=True)

    def _recover_state(self) -> None:
        if not self.state_store:
            return

        queued_jobs = []
        needs_finalization = []
        recovered = self.state_store.load_active_state()
        for request, jobs in recovered:
            self.active_requests[request.request_id] = request
            if not request.preparation_complete:
                # Preparation is not resumable: never accept an incomplete proof graph.
                request.preparation_error = "Coordinator restarted during preparation"
                request.preparation_complete = True
                for job in jobs:
                    if job.job_status in {JobStatus.PREPARING, JobStatus.QUEUED, JobStatus.IN_PROGRESS}:
                        job.job_status = JobStatus.FAILED
                        job.error_message = request.preparation_error
                        job.completed_time = datetime.now(timezone.utc)
                        self.state_store.save_job(job)
                self.state_store.save_request(request)
            for job in jobs:
                if job.job_status == JobStatus.IN_PROGRESS:
                    job.job_status = JobStatus.QUEUED
                    job.started_time = None
                    job.assigned_node_id = None
                    job.lease_id = None
                if job.job_status == JobStatus.QUEUED:
                    queued_jobs.append(job)
            if request.all_jobs_finished() and request.completed_time is None:
                needs_finalization.append(request.request_id)

        self.scheduler.add_jobs(queued_jobs)
        if recovered:
            self.logger.info(
                "Recovered %d requests and %d queued jobs from state store",
                len(recovered),
                len(queued_jobs),
            )

        for request_id in needs_finalization:
            self._schedule_request_finalization(request_id)

    def _new_request(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        simplify_model: bool = False,
        input_shapes: Optional[Dict] = None,
    ) -> InferenceRequest:
        request = InferenceRequest(
            name=name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            split_mode=split_mode,
            ops_per_chunk=ops_per_chunk,
            scheduler=self.scheduler_policy,
            simplify_model=simplify_model,
            input_shapes=input_shapes,
        )
        request.request_status = RequestStatus.PREPARING
        request.preparation_complete = False
        request.preparation_metrics.update(
            scheduler_policy=self.scheduler_policy,
            preparation_dispatch=self.preparation_dispatch,
            witness_preparation_mode=self.request_builder.witness_preparation,
        )
        self._create_request_dirs(request)
        self.active_requests[request.request_id] = request
        if self.state_store:
            self.state_store.save_request(request)
            self.state_store.append_event("request_accepted", request_id=request.request_id)
        return request

    def enqueue_request(self, *args, **kwargs) -> str:
        """Accept and persist an RPC request without waiting for preparation."""
        if not self.preparation_slots.acquire(blocking=False):
            raise PreparationQueueFull("Preparation queue is full; retry later")
        try:
            with self.lock:
                if not self.accepting_requests:
                    raise PreparationQueueFull("Coordinator is shutting down")
                request = self._new_request(*args, **kwargs)
                enqueued = time.perf_counter()
                try:
                    future = self.preparation_pool.submit(self._prepare_request, request, enqueued)
                except Exception as exc:
                    self._fail_preparation(request, exc)
                    raise
                future.add_done_callback(
                    lambda fut, req=request: self._preparation_finished(req, fut)
                )
        except BaseException:
            self.preparation_slots.release()
            raise
        return request.request_id

    def submit_request(self, *args, **kwargs) -> str:
        """Blocking in-process entry point retained for existing callers/tests."""
        with self.lock:
            if not self.accepting_requests:
                raise PreparationQueueFull("Coordinator is shutting down")
            request = self._new_request(*args, **kwargs)
        self._prepare_request(request, time.perf_counter())
        return request.request_id

    def _build_prepared_jobs(self, request, **callbacks):
        return run_preparation(
            request, self.request_builder, self.shared_storage, self.setup_store,
            self.max_retries, self.preparation_rayon_threads, **callbacks,
        )

    def _prepare_request(self, request: InferenceRequest, enqueued: float) -> None:
        started = time.perf_counter()
        request.preparation_metrics["preparation_queue_wait_time(s)"] = started - enqueued
        try:
            if self.preparation_dispatch == "streaming":
                self._prepare_streaming_request(request)
                return

            jobs = self._build_prepared_jobs(request)
            if not jobs:
                raise ValueError("Request contains no proof jobs")
            ready_s = time.perf_counter() - started
            with self.lock:
                request.proof_jobs = jobs
                request.expected_job_count = len(jobs)
                request.preparation_metrics.update({
                    "first_job_ready_time(s)": ready_s,
                    "all_jobs_ready_time(s)": ready_s,
                    "preparation_finished_time(s)": ready_s,
                    "preparation_overlap_time(s)": 0.0,
                })
                request.preparation_complete = True
                self._enqueue_request(request)
        except Exception as exc:
            self._fail_preparation(request, exc)

    def _fail_preparation(self, request: InferenceRequest, exc: BaseException) -> None:
        self.logger.error("Preparation failed for %s: %s", request.request_id, exc)
        with self.lock:
            request.preparation_error = str(exc) or type(exc).__name__
            request.preparation_complete = True
            # Normally a batch failure has no published jobs. Streaming failures
            # drain already released jobs using the existing finalization guard.
            if self.state_store:
                self.state_store.save_request(request)
                self.state_store.append_event(
                    "preparation_finished", request_id=request.request_id,
                    payload={"error": request.preparation_error},
                )
        self._schedule_request_finalization(request.request_id)

    def _preparation_finished(self, request: InferenceRequest, future: Future) -> None:
        try:
            future.result()
        except BaseException as exc:
            self._fail_preparation(request, exc)
        finally:
            self.preparation_slots.release()

    def _prepare_streaming_request(self, request: InferenceRequest) -> None:
        """Publish ready jobs under the scheduler lock; seal before verification."""
        started = time.perf_counter()

        def register_plan(count: int) -> None:
            with self.lock:
                if count < 1 or request.expected_job_count is not None:
                    raise ValueError("Expected a single nonempty preparation plan")
                request.expected_job_count = count
                if self.state_store:
                    self.state_store.save_request(request)

        def publish(job: ProofJob) -> None:
            with self.lock:
                if request.preparation_complete:
                    raise RuntimeError("Cannot publish to a sealed request")
                if request.expected_job_count is None or len(request.proof_jobs) >= request.expected_job_count:
                    raise RuntimeError("Published more jobs than planned")
                job.job_status = JobStatus.QUEUED
                if self.state_store:
                    self.state_store.save_job(job)
                request.proof_jobs.append(job)
                if request.queued_time is None:
                    request.queued_time = datetime.now(timezone.utc)
                    request.request_status = RequestStatus.QUEUED
                    request.preparation_metrics["first_job_ready_time(s)"] = time.perf_counter() - started
                self.scheduler.add_jobs([job])
                if self.state_store:
                    self.state_store.save_request(request)
                    self.state_store.append_event("job_ready", request_id=request.request_id, job_id=job.job_id)

        try:
            self._build_prepared_jobs(request, on_job_ready=publish, on_plan=register_plan)
            if not request.expected_job_count or len(request.proof_jobs) != request.expected_job_count:
                raise RuntimeError("Preparation did not publish every planned job")
        except Exception as exc:
            # Drain already published jobs, but never verify or accept a partial graph.
            self.logger.exception("Preparation failed for request %s", request.request_id)
            request.preparation_error = str(exc) or type(exc).__name__
        finally:
            with self.lock:
                elapsed = time.perf_counter() - started
                if request.preparation_error is None:
                    request.preparation_metrics["all_jobs_ready_time(s)"] = elapsed
                request.preparation_metrics["preparation_finished_time(s)"] = elapsed
                request.preparation_metrics["preparation_overlap_time(s)"] = (
                    max(0.0, (datetime.now(timezone.utc) - request.started_time).total_seconds())
                    if request.started_time else 0.0
                )
                request.preparation_complete = True
                if self.state_store:
                    self.state_store.save_request(request)
                    self.state_store.append_event("preparation_finished", request_id=request.request_id,
                                                  payload={"error": request.preparation_error})
            self._schedule_request_finalization(request.request_id)

    def _create_request_dirs(self, request: InferenceRequest) -> None:
        run_dir = self.runs_dir / request.request_id
        reports_dir = run_dir / "reports"
        run_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        request.run_dir = str(run_dir)
        request.reports_dir = str(reports_dir)

    def get_request_status(self, request_id: str) -> Dict:
        with self.lock:
            request = self.active_requests.get(request_id)
            if request is None:
                return {
                    "request_id": request_id,
                    "scheduler": self.scheduler_policy,
                    "status": "unknown",
                    "message": f"Request {request_id} not found.",
                    "total_jobs": 0,
                    "queued_jobs": 0,
                    "in_progress_jobs": 0,
                    "completed_jobs": 0,
                    "failed_jobs": 0,
                }
            jobs = request.proof_jobs
            return {
                "request_id": request_id,
                "scheduler": self.scheduler_policy,
                "status": request.request_status.value.lower(),
                "message": request.error_message or "",
                "total_jobs": request.expected_job_count if request.expected_job_count is not None else len(jobs),
                "queued_jobs": sum(1 for job in jobs if job.job_status == JobStatus.QUEUED),
                "in_progress_jobs": sum(
                    1 for job in jobs if job.job_status == JobStatus.IN_PROGRESS
                ),
                "completed_jobs": sum(
                    1 for job in jobs if job.job_status == JobStatus.COMPLETED
                ),
                "failed_jobs": sum(1 for job in jobs if job.job_status == JobStatus.FAILED),
            }

    def poll_node(self, node: NodeResources) -> Optional[ProofJob]:
        with self.lock:
            self._record_node_locked(node=node, status="POLLING")
            job = self.scheduler.next_job(node)
            if job is None:
                return None
            self._mark_job_in_progress_locked(job, node.node_id)
            self.logger.info(
                "Assigned job %s to %s | request=%s",
                job.job_id,
                node.node_id,
                job.inference_request_id,
            )
            return job

    def record_node_heartbeat(
        self,
        node_id: str,
        active_job_ids: Optional[list[str]] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self.lock:
            self._record_node_heartbeat_locked(node_id, active_job_ids, status, message)

    def _record_node_heartbeat_locked(
        self,
        node_id: str,
        active_job_ids: Optional[list[str]] = None,
        status: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        previous = self.node_status.get(node_id, {})
        self.node_heartbeats[node_id] = now
        updated = {**previous, "timestamp": now.isoformat()}
        if active_job_ids is not None:
            updated["active_job_ids"] = list(active_job_ids)
        if status is not None:
            updated["status"] = status
        if message is not None:
            updated["message"] = message
        self.node_status[node_id] = updated

    def _record_node_locked(self, node: NodeResources, status: str) -> None:
        now = datetime.now(timezone.utc)
        self.node_heartbeats[node.node_id] = now
        self.node_status[node.node_id] = {
            "status": status,
            "timestamp": now.isoformat(),
            "memory_total_gb": node.memory_total_gb,
            "memory_available_gb": node.memory_available_gb,
            "cpu_total_cores": node.cpu_total_cores,
            "cpu_available_cores": node.cpu_available_cores,
            "active_job_ids": list(node.active_job_ids or []),
        }

    def check_for_dead_nodes(self) -> None:
        timeout_sec = self.heartbeat_config.timeout_sec
        now = datetime.now(timezone.utc)
        with self.lock:
            dead_nodes = [
                node_id
                for node_id, last_seen in self.node_heartbeats.items()
                if (now - last_seen).total_seconds() > timeout_sec
            ]
            for node_id in dead_nodes:
                elapsed_sec = (now - self.node_heartbeats[node_id]).total_seconds()
                self.logger.warning(
                    "Node %s timed out after %.1fs; requeuing its assigned jobs",
                    node_id,
                    elapsed_sec,
                )
                leased_jobs = [
                    job.job_id
                    for job in self.active_jobs.values()
                    if job.assigned_node_id == node_id
                    and job.job_status == JobStatus.IN_PROGRESS
                ]
                for job_id in leased_jobs:
                    self._requeue_active_job_locked(job_id)
                self.node_heartbeats.pop(node_id, None)
                self.node_status.pop(node_id, None)

    def submit_job_result(
        self,
        node_id: str,
        job_id: str,
        status: str,
        lease_id: Optional[str] = None,
        artifact_manifest_ref: Optional[ArtifactRef] = None,
        message: Optional[str] = None,
        perf_metrics: Optional[Dict] = None,
    ) -> bool:
        with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                known_job = next(
                    (item for request in self.active_requests.values()
                     for item in request.proof_jobs if item.job_id == job_id),
                    None,
                )
                if known_job is None and self.state_store:
                    known_job = self.state_store.load_job(job_id)
                return bool(
                    known_job
                    and known_job.job_status in {JobStatus.COMPLETED, JobStatus.FAILED}
                    and known_job.job_status.value == status
                    and known_job.assigned_node_id == node_id
                    and lease_id
                    and known_job.lease_id == lease_id
                    and known_job.artifact_manifest_ref == artifact_manifest_ref
                )
            if job.assigned_node_id != node_id:
                self.logger.error(
                    "Job %s result came from node %s but was assigned to %s",
                    job_id,
                    node_id,
                    job.assigned_node_id,
                )
                return False
            if not lease_id or lease_id != job.lease_id:
                self.logger.error(
                    "Rejected stale result for job %s: assignment token %s, expected %s",
                    job_id,
                    lease_id,
                    job.lease_id,
                )
                return False
            try:
                job_status = JobStatus(status)
            except ValueError:
                self.logger.error("Invalid job status '%s' for job %s", status, job_id)
                return False

            if job_status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
                self.logger.error("Non-terminal result status '%s' for job %s", status, job_id)
                return False

            job.completed_time = datetime.now(timezone.utc)
            job.job_status = job_status
            job.artifact_manifest_ref = artifact_manifest_ref

            if job.job_status == JobStatus.FAILED:
                parent = self.active_requests.get(job.inference_request_id)
                if parent:
                    attempt = copy.copy(job)
                    attempt.error_message = message
                    try:
                        write_failed_attempt_report(attempt, perf_metrics, parent.reports_dir)
                    except Exception:
                        self.logger.exception("Failed to write attempt report")
                should_retry = self._handle_failed_job_locked(job, message)
                if should_retry:
                    self._record_node_heartbeat_locked(
                        node_id=node_id,
                        status=f"JOB_{job_status.value}",
                    )
                    return True
            else:
                self._complete_job_locked(job)
            self._record_node_heartbeat_locked(
                node_id=node_id,
                status=f"JOB_{job_status.value}",
            )
            self.pending_job_reports.add(job.job_id)

        try:
            self._finalize_job(job, perf_metrics=perf_metrics)
        finally:
            with self.lock:
                self.pending_job_reports.discard(job.job_id)
            self._schedule_request_finalization(job.inference_request_id)
        return True

    def _enqueue_request(self, request: InferenceRequest) -> None:
        request.queued_time = datetime.now(timezone.utc)
        request.request_status = RequestStatus.QUEUED
        for job in request.proof_jobs:
            job.job_status = JobStatus.QUEUED
        with self.lock:
            self.active_requests[request.request_id] = request
            if self.state_store:
                self.state_store.save_request(request)
                self.state_store.save_jobs(request.proof_jobs)
                self.state_store.append_event(
                    "request_queued",
                    request_id=request.request_id,
                )
            self.scheduler.add_jobs(request.proof_jobs)

    def _mark_job_in_progress_locked(self, job: ProofJob, node_id: str) -> None:
        job.job_status = JobStatus.IN_PROGRESS
        job.started_time = datetime.now(timezone.utc)
        job.assigned_node_id = node_id
        job.lease_id = f"{job.job_id}-{uuid.uuid4().hex[:8]}"
        self.active_jobs[job.job_id] = job
        parent_request = self.active_requests.get(job.inference_request_id)
        if parent_request and parent_request.started_time is None:
            parent_request.started_time = datetime.now(timezone.utc)
            parent_request.request_status = RequestStatus.IN_PROGRESS
            if self.state_store:
                self.state_store.save_request(parent_request)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event(
                "job_leased",
                request_id=job.inference_request_id,
                job_id=job.job_id,
                payload={"node_id": node_id, "lease_id": job.lease_id},
            )

    def _requeue_active_job_locked(self, job_id: str) -> None:
        job = self.active_jobs.get(job_id)
        if not job or job.job_status != JobStatus.IN_PROGRESS:
            return
        job.job_status = JobStatus.QUEUED
        job.started_time = None
        job.completed_time = None
        job.assigned_node_id = None
        job.lease_id = None
        self.scheduler.requeue(job)
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event(
                "job_requeued",
                request_id=job.inference_request_id,
                job_id=job.job_id,
            )
        self.logger.info("Requeued job %s after node failure", job.job_id)

    def _complete_job_locked(self, job: ProofJob) -> None:
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event(
                "job_completed",
                request_id=job.inference_request_id,
                job_id=job.job_id,
            )

        parent_request = self.active_requests.get(job.inference_request_id)
        if not parent_request:
            self.logger.info("Job %s completed", job.job_id)
            return

        total = parent_request.expected_job_count or len(parent_request.proof_jobs)
        completed = sum(
            1 for item in parent_request.proof_jobs if item.job_status == JobStatus.COMPLETED
        )
        running = sum(
            1 for item in parent_request.proof_jobs if item.job_status == JobStatus.IN_PROGRESS
        )
        queued = sum(
            1 for item in parent_request.proof_jobs if item.job_status == JobStatus.QUEUED
        )
        failed = sum(
            1 for item in parent_request.proof_jobs if item.job_status == JobStatus.FAILED
        )
        percent = (100.0 * completed / total) if total else 0.0
        self.logger.info(
            "Completed job %s | request=%s | progress=%d/%d (%.1f%%) | running=%d queued=%d failed=%d",
            job.job_id,
            parent_request.request_id,
            completed,
            total,
            percent,
            running,
            queued,
            failed,
        )

    def _handle_failed_job_locked(
        self,
        job: ProofJob,
        message: Optional[str],
    ) -> bool:
        job.error_message = message
        job.retry_count += 1
        if job.retry_count <= job.max_retries:
            self.logger.warning(
                "Job %s failed; retry %d/%d",
                job.job_id,
                job.retry_count,
                job.max_retries,
            )
            job.job_status = JobStatus.QUEUED
            job.started_time = None
            job.completed_time = None
            job.assigned_node_id = None
            job.lease_id = None
            job.artifact_manifest_ref = None
            job.error_message = None
            self.active_jobs.pop(job.job_id, None)
            self.scheduler.requeue(job)
            if self.state_store:
                self.state_store.save_job(job)
                self.state_store.append_event(
                    "job_retry",
                    request_id=job.inference_request_id,
                    job_id=job.job_id,
                )
            return True

        job.error_message = message or "Job failed permanently after retries."
        self.active_jobs.pop(job.job_id, None)
        if self.state_store:
            self.state_store.save_job(job)
            self.state_store.append_event(
                "job_failed",
                request_id=job.inference_request_id,
                job_id=job.job_id,
            )
        self.logger.error("Job %s failed permanently: %s", job.job_id, message)
        return False

    def _finalize_job(
        self,
        job: ProofJob,
        perf_metrics: Optional[Dict] = None,
    ) -> None:
        parent_request = self.active_requests.get(job.inference_request_id)
        if not parent_request:
            return

        profile_key = job.profiling_file_path or job.job_id
        with self.lock:
            profile_lock = self.profile_locks.setdefault(profile_key, threading.Lock())
        with profile_lock:
            self._write_job_report_and_profile(job, parent_request, perf_metrics)

    def _write_job_report_and_profile(
        self, job: ProofJob, parent_request: InferenceRequest,
        perf_metrics: Optional[Dict],
    ) -> None:

        job_report = write_job_report(
            job,
            perf_metrics=perf_metrics,
            out_dir=parent_request.reports_dir,
        )
        if job.artifact_manifest_ref:
            job_report["artifact_manifest_backend"] = job.artifact_manifest_ref.backend
            job_report["artifact_manifest_uri"] = job.artifact_manifest_ref.uri

        update_history = self.setup_store.resource_profiles_update_history
        if update_history is None:
            update_history = self.setup_store.resource_profiles_enabled
        if job.job_status == JobStatus.COMPLETED and job.profiling_file_path and update_history:
            save_json(
                job_report,
                job.profiling_file_path,
                storage_type=self.setup_store.backend,
                s3_bucket=self.setup_store.s3_bucket,
            )
            if not self.setup_store.resource_profiles_enabled:
                return
            estimates = get_profile_estimates(job_report)
            with self.lock:
                for request in self.active_requests.values():
                    for queued_job in request.proof_jobs:
                        if (queued_job.job_status == JobStatus.QUEUED
                                and queued_job.profiling_file_path == job.profiling_file_path
                                and queued_job.artifact_key == job.artifact_key):
                            queued_job.profiling_data = job_report
                            queued_job.predicted_duration_s = float(
                                estimates.get("job_runtime_s") or 0.0
                            )
                            queued_job.predicted_peak_memory_gb = max_stage_memory_gb(job_report)
                            if self.state_store:
                                self.state_store.save_job(queued_job)

    def _schedule_request_finalization(self, request_id: str) -> None:
        with self.lock:
            request = self.active_requests.get(request_id)
            if not request:
                return
            if not request.all_jobs_finished() or request.completed_time is not None:
                return
            if any(job.job_id in self.pending_job_reports for job in request.proof_jobs):
                return
            if request_id in self.finalizing_requests:
                return

            self.finalizing_requests.add(request_id)
            if request.preparation_error is not None or request.any_job_failed():
                request.request_status = RequestStatus.IN_PROGRESS
            elif self.verifier:
                request.request_status = RequestStatus.VERIFYING
                if self.state_store:
                    self.state_store.save_request(request)
                    self.state_store.append_event(
                        "request_verifying",
                        request_id=request.request_id,
                    )

            future = self.finalization_pool.submit(
                self._finalize_request,
                request_id,
            )
            future.add_done_callback(
                lambda fut, rid=request_id: self._finalization_finished(rid, fut)
            )

    def _finalization_finished(self, request_id: str, future: Future) -> None:
        try:
            future.result()
        except Exception as exc:
            self.logger.error(
                "Request finalization crashed for %s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            with self.lock:
                request = self.active_requests.get(request_id)
                if request and request.completed_time is None:
                    request.request_status = RequestStatus.FAILED
                    request.error_message = f"Request finalization failed: {exc}"
                    request.completed_time = datetime.now(timezone.utc)
                    if self.state_store:
                        self.state_store.save_request(request)
        finally:
            with self.lock:
                self.finalizing_requests.discard(request_id)

    def _finalize_request(self, request_id: str) -> None:
        with self.lock:
            parent_request = self.active_requests.get(request_id)
            if not parent_request:
                return

        verification_report: Optional[Dict] = None
        if parent_request.preparation_error is not None:
            final_status = RequestStatus.FAILED
            error_message = f"Preparation failed: {parent_request.preparation_error}"
        elif parent_request.any_job_failed():
            failed_count = sum(
                1
                for job in parent_request.proof_jobs
                if job.job_status == JobStatus.FAILED
            )
            final_status = RequestStatus.FAILED
            error_message = "One or more sub-jobs failed."
            self.logger.error(
                "Request %s failed. %d jobs failed.",
                parent_request.request_id,
                failed_count,
            )
        else:
            if self.verifier:
                self.logger.info("Request %s | verifying %d proofs", parent_request.request_id, len(parent_request.proof_jobs))
                verify_started = time.perf_counter()
                verification_report = self._verify_request_if_enabled(parent_request)
                parent_request.preparation_metrics["verification_time(s)"] = (
                    time.perf_counter() - verify_started
                )
                self.logger.info("Request %s | verification %s | %.2fs", parent_request.request_id,
                                 "passed" if verification_report and verification_report.get("valid") else "failed",
                                 parent_request.preparation_metrics["verification_time(s)"])
            else:
                exact = parent_request.preparation_metrics.get("witness_preparation_mode") == "upfront"
                verification_report = {
                    "enabled": False, "valid": not exact,
                    "error": "Upfront requests require verification" if exact else "",
                }

            if verification_report and not verification_report.get("valid", True):
                final_status = RequestStatus.FAILED
                error_message = "Proof verification failed."
                self.logger.error(
                    "Request %s failed verification.",
                    parent_request.request_id,
                )
            else:
                final_status = RequestStatus.COMPLETED
                error_message = None
                self.logger.info("Request %s completed.", parent_request.request_id)

        with self.lock:
            parent_request.request_status = final_status
            parent_request.error_message = error_message
            parent_request.completed_time = datetime.now(timezone.utc)
            if self.state_store:
                self.state_store.save_request(parent_request)
                self.state_store.append_event(
                    "request_finished",
                    request_id=parent_request.request_id,
                    payload={
                        "verification": verification_report or {"enabled": False},
                    },
                )

        try:
            write_request_report(
                parent_request,
                out_dir=parent_request.reports_dir,
            )
            transfer_overhead = write_transfer_overhead_report(
                parent_request.reports_dir
            )
            if self.verification_config.write_request_manifest:
                write_request_artifact_manifest(
                    request=parent_request,
                    out_dir=parent_request.reports_dir,
                    verification=verification_report or {"enabled": False},
                    aggregate_metrics=transfer_overhead,
                )
        except Exception:
            self.logger.exception(
                "Failed to write final reports for request %s",
                parent_request.request_id,
            )

    def _verify_request_if_enabled(
        self,
        request: InferenceRequest,
    ) -> Optional[Dict]:
        if not self.verifier:
            return {"enabled": False, "valid": True}
        try:
            report = run_verification(request, self.verifier, self.preparation_rayon_threads)
            if request.reports_dir:
                save_json(
                    report,
                    str(Path(request.reports_dir) / "verification_report.json"),
                    storage_type="filesystem",
                )
            return report
        except Exception as exc:
            self.logger.error(
                "Verification failed for request %s: %s",
                request.request_id,
                exc,
                exc_info=True,
            )
            return {"enabled": True, "valid": False, "error": str(exc)}

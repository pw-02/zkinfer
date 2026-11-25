from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.models import InferenceRequest, JobStatus, ProofJob, RequestStatus


TERMINAL_REQUEST_STATES = {
    RequestStatus.COMPLETED.value,
    RequestStatus.FAILED.value,
    RequestStatus.CANCELLED.value,
}


def _dt_to_str(value):
    return value.isoformat() if value else None


def _dt_from_str(value):
    return datetime.fromisoformat(value) if value else None


def _artifact_to_dict(ref: ArtifactRef | None) -> Dict:
    return ref.to_dict() if ref else ArtifactRef.empty().to_dict()


def _artifact_from_dict(data: Dict | None) -> ArtifactRef:
    return ArtifactRef.from_dict(data)


class CoordinatorStateStore:
    """SQLite-backed coordinator state with in-place schema migration."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    @classmethod
    def _ensure_column(
        cls,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        if column not in cls._table_columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    assigned_node_id TEXT,
                    lease_id TEXT,
                    lease_updated_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    request_id TEXT,
                    job_id TEXT,
                    payload_json TEXT
                )
                """
            )

            # Older databases used assigned_worker_id and lacked lease columns.
            # Add the current columns without destroying prior experiment state.
            self._ensure_column(conn, "jobs", "assigned_node_id", "TEXT")
            self._ensure_column(conn, "jobs", "lease_id", "TEXT")
            self._ensure_column(conn, "jobs", "lease_updated_at", "TEXT")
            self._ensure_column(conn, "jobs", "updated_at", "TEXT")

            columns = self._table_columns(conn, "jobs")
            if "assigned_worker_id" in columns:
                conn.execute(
                    """
                    UPDATE jobs
                    SET assigned_node_id = COALESCE(assigned_node_id, assigned_worker_id)
                    WHERE assigned_worker_id IS NOT NULL
                    """
                )

    def save_request(self, request: InferenceRequest) -> None:
        payload = self._request_to_dict(request)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO requests(request_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (request.request_id, request.request_status.value, json.dumps(payload), now),
            )

    def save_jobs(self, jobs: Iterable[ProofJob]) -> None:
        for job in jobs:
            self.save_job(job)

    def save_job(self, job: ProofJob) -> None:
        payload = self._job_to_dict(job)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, request_id, status, payload_json,
                    assigned_node_id, lease_id, lease_updated_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    request_id=excluded.request_id,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    assigned_node_id=excluded.assigned_node_id,
                    lease_id=excluded.lease_id,
                    lease_updated_at=excluded.lease_updated_at,
                    updated_at=excluded.updated_at
                """,
                (
                    job.job_id,
                    job.inference_request_id,
                    job.job_status.value,
                    json.dumps(payload),
                    job.assigned_node_id,
                    job.lease_id,
                    _dt_to_str(job.started_time),
                    now,
                ),
            )

    def append_event(
        self,
        event_type: str,
        request_id: str = "",
        job_id: str = "",
        payload: Dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(created_at, event_type, request_id, job_id, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    event_type,
                    request_id or None,
                    job_id or None,
                    json.dumps(payload or {}),
                ),
            )

    def load_active_state(self) -> List[Tuple[InferenceRequest, List[ProofJob]]]:
        with self._connect() as conn:
            req_rows = conn.execute(
                "SELECT * FROM requests WHERE status NOT IN (?, ?, ?)",
                tuple(TERMINAL_REQUEST_STATES),
            ).fetchall()
            recovered = []
            for req_row in req_rows:
                request = self._request_from_dict(json.loads(req_row["payload_json"]))
                job_rows = conn.execute(
                    "SELECT * FROM jobs WHERE request_id = ? ORDER BY updated_at ASC",
                    (request.request_id,),
                ).fetchall()
                jobs = []
                for job_row in job_rows:
                    job = self._job_from_dict(json.loads(job_row["payload_json"]))
                    if job.job_status == JobStatus.IN_PROGRESS:
                        job.job_status = JobStatus.QUEUED
                        job.started_time = None
                        job.assigned_node_id = None
                        job.lease_id = None
                    jobs.append(job)
                request.proof_jobs = jobs
                if request.request_status == RequestStatus.VERIFYING:
                    request.request_status = RequestStatus.IN_PROGRESS
                    request.completed_time = None
                recovered.append((request, jobs))
            return recovered

    def load_job(self, job_id: str) -> ProofJob | None:
        """Read a terminal job too, for completion retries after a restart."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._job_from_dict(json.loads(row["payload_json"])) if row else None

    @staticmethod
    def _request_to_dict(request: InferenceRequest) -> Dict:
        return {
            "name": request.name,
            "onnx_model_path": request.onnx_model_path,
            "input_data_path": request.input_data_path,
            "split_mode": request.split_mode,
            "ops_per_chunk": request.ops_per_chunk,
            "scheduler": request.scheduler,
            "simplify_model": request.simplify_model,
            "input_shapes": request.input_shapes,
            "request_id": request.request_id,
            "preparation_metrics": dict(request.preparation_metrics),
            "preparation_complete": request.preparation_complete,
            "expected_job_count": request.expected_job_count,
            "preparation_error": request.preparation_error,
            "created_time": _dt_to_str(request.created_time),
            "queued_time": _dt_to_str(request.queued_time),
            "started_time": _dt_to_str(request.started_time),
            "completed_time": _dt_to_str(request.completed_time),
            "error_message": request.error_message,
            "request_status": request.request_status.value,
            "run_dir": request.run_dir,
            "logs_dir": request.logs_dir,
            "reports_dir": request.reports_dir,
        }

    @staticmethod
    def _request_from_dict(data: Dict) -> InferenceRequest:
        request = InferenceRequest(
            name=data["name"],
            onnx_model_path=data["onnx_model_path"],
            input_data_path=data["input_data_path"],
            split_mode=data["split_mode"],
            ops_per_chunk=int(data["ops_per_chunk"]),
            scheduler=data["scheduler"],
            simplify_model=bool(data.get("simplify_model", False)),
            input_shapes=data.get("input_shapes"),
        )
        request.request_id = data["request_id"]
        request.preparation_metrics = data.get("preparation_metrics", {})
        request.preparation_complete = data.get("preparation_complete", True)
        request.expected_job_count = data.get("expected_job_count")
        request.preparation_error = data.get("preparation_error")
        request.created_time = _dt_from_str(data.get("created_time")) or datetime.now(timezone.utc)
        request.queued_time = _dt_from_str(data.get("queued_time"))
        request.started_time = _dt_from_str(data.get("started_time"))
        request.completed_time = _dt_from_str(data.get("completed_time"))
        request.error_message = data.get("error_message")
        request.request_status = RequestStatus(data.get("request_status", RequestStatus.QUEUED.value))
        request.run_dir = data.get("run_dir")
        request.logs_dir = data.get("logs_dir")
        request.reports_dir = data.get("reports_dir")
        return request

    @staticmethod
    def _job_to_dict(job: ProofJob) -> Dict:
        return {
            "inference_request_name": job.inference_request_name,
            "model_name": job.model_name,
            "inference_request_id": job.inference_request_id,
            "model_ref": _artifact_to_dict(job.model_ref),
            "input_ref": _artifact_to_dict(job.input_ref),
            "setup_cache_ref": _artifact_to_dict(job.setup_cache_ref),
            "output_prefix_ref": _artifact_to_dict(job.output_prefix_ref),
            "profiling_file_path": job.profiling_file_path,
            "model_write_time": job.model_write_time,
            "input_write_time": job.input_write_time,
            "profiling_data": job.profiling_data,
            "predicted_duration_s": job.predicted_duration_s,
            "predicted_peak_memory_gb": job.predicted_peak_memory_gb,
            "artifact_key": job.artifact_key,
            "max_retries": job.max_retries,
            "job_id": job.job_id,
            "job_name": job.job_name,
            "job_status": job.job_status.value,
            "queued_time": _dt_to_str(job.queued_time),
            "started_time": _dt_to_str(job.started_time),
            "completed_time": _dt_to_str(job.completed_time),
            "artifact_manifest_ref": _artifact_to_dict(job.artifact_manifest_ref),
            "assigned_node_id": job.assigned_node_id,
            "lease_id": job.lease_id,
            "error_message": job.error_message,
            "retry_count": job.retry_count,
            "parent_model_hash": job.parent_model_hash,
            "model_hash": job.model_hash,
        }

    @staticmethod
    def _job_from_dict(data: Dict) -> ProofJob:
        job = ProofJob(
            inference_request_name=data["inference_request_name"],
            model_name=data["model_name"],
            inference_request_id=data["inference_request_id"],
            model_ref=_artifact_from_dict(data.get("model_ref")),
            input_ref=_artifact_from_dict(data.get("input_ref")),
            setup_cache_ref=_artifact_from_dict(data.get("setup_cache_ref")),
            output_prefix_ref=_artifact_from_dict(data.get("output_prefix_ref")),
            profiling_file_path=data.get("profiling_file_path"),
            model_write_time=float(data.get("model_write_time") or 0.0),
            input_write_time=float(data.get("input_write_time") or 0.0),
            profiling_data=data.get("profiling_data") or {},
            predicted_duration_s=float(data.get("predicted_duration_s") or 0.0),
            predicted_peak_memory_gb=float(data.get("predicted_peak_memory_gb") or 0.0),
            artifact_key=data.get("artifact_key") or "",
            max_retries=int(data.get("max_retries") or 0),
            parent_model_hash=data.get("parent_model_hash"),
            model_hash=data.get("model_hash"),
        )
        job.job_id = data["job_id"]
        job.job_name = data.get("job_name") or f"{job.inference_request_name}_{job.model_name}"
        job.job_status = JobStatus(data.get("job_status", JobStatus.QUEUED.value))
        job.queued_time = _dt_from_str(data.get("queued_time")) or datetime.now(timezone.utc)
        job.started_time = _dt_from_str(data.get("started_time"))
        job.completed_time = _dt_from_str(data.get("completed_time"))
        manifest_ref = _artifact_from_dict(data.get("artifact_manifest_ref"))
        job.artifact_manifest_ref = None if manifest_ref.is_empty else manifest_ref
        job.assigned_node_id = data.get("assigned_node_id")
        job.lease_id = data.get("lease_id")
        job.error_message = data.get("error_message")
        job.retry_count = int(data.get("retry_count") or 0)
        return job


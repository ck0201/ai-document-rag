"""
Asynchronous Ingestion Job Tracker.

Manages background document processing states:
PENDING -> PROCESSING -> INDEXED | FAILED
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Dict, Optional
import uuid


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class IngestionJob:
    job_id: str
    filename: str
    file_path: str
    status: JobStatus = JobStatus.PENDING
    chunks_count: int = 0
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


class JobManager:
    """Thread-safe in-memory job repository for tracking async ingestion."""

    def __init__(self):
        self._jobs: Dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def create_job(self, filename: str, file_path: str) -> IngestionJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = IngestionJob(job_id=job_id, filename=filename, file_path=file_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        chunks_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                job.chunks_count = chunks_count
                job.error_message = error_message
                if status in (JobStatus.INDEXED, JobStatus.FAILED):
                    job.completed_at = datetime.now(timezone.utc).isoformat()

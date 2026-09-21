"""In-process background jobs with progress, cancellation and SSE-friendly state.

Tracking and export both run on a single worker thread each, serialised behind one
semaphore: a GPU cannot run two trackers at once without thrashing VRAM, so
juggling them in one process is simpler than a broker and enough for a studio
tool driven by one editor.
"""
from __future__ import annotations

import logging
import threading
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from .storage import new_id, utcnow

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class JobCancelled(Exception):
    """Raised inside a job body when the user cancels it."""


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""
    video_id: str | None = None
    result: dict[str, Any] | None = None
    cancel_requested: bool = False
    _traceback: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "video_id": self.video_id,
            "result": self.result,
        }

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_STATUSES


JobBody = Callable[["JobContext"], dict[str, Any] | None]


class JobContext:
    """Handed to a job body: report progress and cooperatively check cancellation."""

    def __init__(self, manager: "JobManager", job: Job):
        self._manager = manager
        self.job = job

    @property
    def video_id(self) -> str | None:
        return self.job.video_id

    def progress(self, fraction: float, message: str = "") -> None:
        self._manager._update(
            self.job.id,
            progress=max(0.0, min(1.0, float(fraction))),
            **({"message": message} if message else {}),
        )

    def check_cancelled(self) -> None:
        if self.job.cancel_requested:
            raise JobCancelled()

    def note(self, message: str) -> None:
        self._manager._update(self.job.id, message=message)


class JobManager:
    def __init__(self, history: int = 200):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._history = history
        # Heavy jobs (tracking, exports) run one at a time.
        self._gate = threading.Semaphore(1)

    # ------------------------------------------------------------------- state
    def create(self, kind: str, *, video_id: str | None = None) -> Job:
        job = Job(
            id=new_id(),
            kind=kind,
            created_at=utcnow(),
            updated_at=utcnow(),
            video_id=video_id,
        )
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._history:
                self._jobs.popitem(last=False)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, video_id: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if video_id is not None:
            jobs = [job for job in jobs if job.video_id == video_id]
        return jobs

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.done:
                return False
            job.cancel_requested = True
            job.message = "cancelling..."
            job.updated_at = utcnow()
            return True

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = utcnow()

    # ------------------------------------------------------------------ running
    def submit(self, job: Job, body: JobBody, *, exclusive: bool = True) -> Job:
        thread = threading.Thread(
            target=self._worker, args=(job, body, exclusive), daemon=True,
            name=f"job-{job.kind}-{job.id}",
        )
        thread.start()
        return job

    def _worker(self, job: Job, body: JobBody, exclusive: bool) -> None:
        acquired = False
        try:
            if exclusive:
                self._gate.acquire()
                acquired = True
            if job.cancel_requested:
                raise JobCancelled()
            self._update(job.id, status="running", message="starting")

            result = body(JobContext(self, job))
            if job.cancel_requested:
                raise JobCancelled()
            self._update(
                job.id, status="succeeded", progress=1.0, result=result or {}, message="done",
            )
        except JobCancelled:
            self._update(job.id, status="cancelled", message="cancelled")
        except Exception as exc:  # noqa: BLE001 - any failure must land on the job
            logger.exception("job %s (%s) failed", job.id, job.kind)
            self._update(
                job.id,
                status="failed",
                message="failed",
                error=f"{type(exc).__name__}: {exc}",
                _traceback=traceback.format_exc(),
            )
        finally:
            if acquired:
                self._gate.release()


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager()


def reset_job_manager() -> None:
    get_job_manager.cache_clear()

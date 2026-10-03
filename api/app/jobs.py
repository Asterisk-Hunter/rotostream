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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from .storage import new_id, utcnow

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class JobCapacityError(RuntimeError):
    """The bounded worker queue cannot accept more work."""


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
    def __init__(self, history: int = 200, max_pending: int = 16):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._history = history
        self._max_pending = max_pending
        self._closed = False
        self._submitted: set[str] = set()
        self._outstanding: set[str] = set()
        self._callbacks: dict[str, Callable[[Job], None]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rotostream-job")
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
            reserved = {key for key, item in self._jobs.items() if not item.done} | self._outstanding
            if self._closed or len(reserved) >= self._max_pending:
                raise JobCapacityError("worker queue is full; retry after current jobs finish")
            self._jobs[job.id] = job
            self._prune()
        return job

    def _prune(self) -> None:
        completed = [key for key, job in self._jobs.items() if job.done and key not in self._outstanding]
        for key in completed[:-self._history]:
            self._jobs.pop(key, None)
            self._submitted.discard(key)

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
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "cancelled"
                self._notify_done(job)
            else:
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
    def submit(self, job: Job, body: JobBody, *, exclusive: bool = True,
               on_done: Callable[[Job], None] | None = None) -> Job:
        with self._lock:
            if job.id in self._submitted:
                raise ValueError("job has already been submitted")
            if self._closed:
                raise JobCapacityError("worker is shutting down")
            self._submitted.add(job.id)
            self._outstanding.add(job.id)
            if on_done:
                self._callbacks[job.id] = on_done
            self._executor.submit(self._worker, job, body, exclusive)
        return job

    def _notify_done(self, job: Job) -> None:
        with self._lock:
            callback = self._callbacks.pop(job.id, None)
        if callback:
            try:
                callback(job)
            except Exception:
                logger.exception("job %s completion cleanup failed", job.id)

    @contextmanager
    def compute_slot(self):
        """Serialize interactive inference with background GPU work."""
        with self._gate:
            yield

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for job in list(self._jobs.values()):
                if not job.done:
                    self.cancel(job.id)
        self._executor.shutdown(wait=True)

    def _worker(self, job: Job, body: JobBody, exclusive: bool) -> None:
        acquired = False
        try:
            if job.done:
                return
            if exclusive:
                while not self._gate.acquire(timeout=0.1):
                    if job.cancel_requested:
                        raise JobCancelled()
                acquired = True
            with self._lock:
                if job.cancel_requested:
                    raise JobCancelled()
                self._update(job.id, status="running", message="starting")

            result = body(JobContext(self, job))
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
            self._notify_done(job)
            with self._lock:
                self._outstanding.discard(job.id)
                self._prune()


@lru_cache
def get_job_manager() -> JobManager:
    from .settings import get_settings
    settings = get_settings()
    return JobManager(history=settings.job_history, max_pending=settings.max_pending_jobs)


def reset_job_manager() -> None:
    if get_job_manager.cache_info().currsize:
        get_job_manager().shutdown()
    get_job_manager.cache_clear()

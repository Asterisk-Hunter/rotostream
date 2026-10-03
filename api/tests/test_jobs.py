"""Deterministic worker admission, cancellation and compute isolation checks."""
from __future__ import annotations

import threading
import time

import pytest

from app.jobs import JobCapacityError, JobManager


def wait_done(job):
    deadline = time.monotonic() + 3
    while not job.done and time.monotonic() < deadline:
        time.sleep(0.005)
    assert job.done


def test_queue_is_bounded_and_cancelled_waiters_do_not_run():
    manager = JobManager(history=1, max_pending=2)
    entered, release = threading.Event(), threading.Event()
    calls = []
    completed = []

    def block(ctx):
        entered.set()
        assert release.wait(3)
        return {"committed": True}

    try:
        running = manager.create("track")
        manager.submit(running, block)
        assert entered.wait(3)
        queued = manager.create("export")
        manager.submit(queued, lambda ctx: calls.append("ran"),
                       on_done=lambda job: completed.append(job.status))
        with pytest.raises(JobCapacityError):
            manager.create("overflow")
        assert manager.cancel(queued.id)
        assert queued.status == "cancelled"
        assert completed == ["cancelled"]
        # Cancelled executor tasks still occupy capacity until drained. An
        # attacker cannot grow its internal queue by creating/cancelling jobs.
        with pytest.raises(JobCapacityError):
            manager.create("overflow")
        assert manager.get(running.id) is running
        release.set()
        wait_done(running)
    finally:
        release.set()
        manager.shutdown()
    assert calls == []
    assert completed == ["cancelled"]
    assert running.status == "succeeded"


def test_active_jobs_survive_completed_history_eviction():
    manager = JobManager(history=1, max_pending=3)
    try:
        active = manager.create("reserved")
        for _ in range(4):
            job = manager.create("short")
            manager.submit(job, lambda ctx: {})
            wait_done(job)
        assert manager.get(active.id) is active
        assert len(manager.list()) <= 2
    finally:
        manager.shutdown()


def test_preview_compute_slot_excludes_background_jobs():
    manager = JobManager()
    entered = threading.Event()
    try:
        with manager.compute_slot():
            job = manager.create("track")
            manager.submit(job, lambda ctx: entered.set())
            assert not entered.wait(0.05)
            assert job.status == "queued"
        assert entered.wait(3)
        wait_done(job)
    finally:
        manager.shutdown()


def test_worker_failures_are_observable_and_allow_the_next_job():
    manager = JobManager()
    try:
        def fail(ctx):
            raise RuntimeError("fixture failure")
        bad = manager.create("bad")
        manager.submit(bad, fail)
        wait_done(bad)
        good = manager.create("good")
        manager.submit(good, lambda ctx: {"value": 1})
        wait_done(good)
        assert bad.status == "failed" and "fixture failure" in bad.error
        assert good.status == "succeeded" and good.result == {"value": 1}
    finally:
        manager.shutdown()

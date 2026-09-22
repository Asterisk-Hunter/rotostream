"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Job, TrackResult } from "@/lib/types";

export interface JobState {
  job: Job<TrackResult> | null;
  error: string | null;
  /** True while the job is queued or running. */
  active: boolean;
  /** True once the job reached a terminal state. */
  done: boolean;
}

interface Snapshot {
  jobId: string;
  job: Job<TrackResult>;
  error: string | null;
}

/**
 * Subscribe to a job's progress.
 *
 * The API exposes `GET /api/jobs/{id}/events` as SSE. If EventSource is
 * unavailable or the stream drops, this falls back to polling, so progress always
 * eventually converges.
 *
 * State is stored together with the job id it belongs to rather than being reset
 * inside the effect: that keeps the effect body free of synchronous state writes
 * (which React flags as cascading renders) and makes a stale job impossible to
 * read after the id changes.
 */
export function useJob(jobId: string | null, pollMs = 400): JobState {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const record = (job: Job<TrackResult>, error: string | null) => {
      if (disposed) return;
      setSnapshot({ jobId, job, error });
    };

    const pull = async () => {
      try {
        const next = await api.getJob(jobId);
        record(next, null);
        if (next.status === "queued" || next.status === "running") {
          timer = setTimeout(() => void pull(), pollMs);
        }
      } catch (cause) {
        if (disposed) return;
        setSnapshot((current) =>
          current && current.jobId === jobId
            ? { ...current, error: cause instanceof Error ? cause.message : String(cause) }
            : current,
        );
      }
    };

    const source =
      typeof EventSource === "undefined" ? null : new EventSource(api.eventsUrl(jobId));
    if (source) {
      source.onmessage = (event) => {
        try {
          record(JSON.parse(event.data) as Job<TrackResult>, null);
        } catch {
          /* ignore malformed frames; the polling fallback will cover it */
        }
      };
      source.addEventListener("done", () => source.close());
      source.onerror = () => {
        source.close();
        void pull();
      };
    }

    void pull();

    return () => {
      disposed = true;
      source?.close();
      if (timer) clearTimeout(timer);
    };
  }, [jobId, pollMs]);

  // Only ever expose the snapshot that belongs to the requested job.
  const current = snapshot && snapshot.jobId === jobId ? snapshot : null;
  const job = current?.job ?? null;
  const active = job?.status === "queued" || job?.status === "running";

  return {
    job,
    error: current?.error ?? null,
    active,
    done: job ? !active : false,
  };
}

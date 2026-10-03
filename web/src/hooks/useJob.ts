"use client";

import { useEffect, useState } from "react";

import { ApiError, api } from "@/lib/api";
import type { Job, TrackResult } from "@/lib/types";
import { monitorJob, parseJobMessage } from "@/lib/jobMonitor";

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
  job: Job<TrackResult> | null;
  error: string | null;
  stopped?: boolean;
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
export function useJob(jobId: string | null, pollMs = 1000): JobState {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  useEffect(() => {
    if (!jobId) return;

    return monitorJob({
      load: () => api.getJob(jobId),
      publish: (next) => setSnapshot({ jobId, ...next }),
      pollMs,
      shouldRetry: (cause) => !(cause instanceof ApiError && [404, 410].includes(cause.status)),
      subscribe: typeof EventSource === "undefined" ? undefined : (receive) => {
        const source = new EventSource(api.eventsUrl(jobId));
        source.onmessage = (event) => {
          const job = parseJobMessage(event.data, jobId);
          if (job) receive(job);
        };
        source.addEventListener("done", () => source.close());
        source.onerror = () => source.close();
        return () => source.close();
      },
    });
  }, [jobId, pollMs]);

  // Only ever expose the snapshot that belongs to the requested job.
  const current = snapshot && snapshot.jobId === jobId ? snapshot : null;
  const job = current?.job ?? null;
  const active = Boolean(jobId && !current?.stopped && (!job || job.status === "queued" || job.status === "running"));

  return {
    job,
    error: current?.error ?? null,
    active,
    done: job ? !active : false,
  };
}

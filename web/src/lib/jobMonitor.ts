import type { Job, TrackResult } from "./types";

export type MonitoredJob = Job<TrackResult>;
export interface MonitorSnapshot {
  job: MonitoredJob | null;
  error: string | null;
  stopped?: boolean;
}

export function isActiveJob(job: MonitoredJob): boolean {
  return job.status === "queued" || job.status === "running";
}

/** Validate progress messages before allowing them to affect editor state. */
export function parseJobMessage(data: string, jobId: string): MonitoredJob | null {
  try {
    const value: unknown = JSON.parse(data);
    if (!value || typeof value !== "object") return null;
    const job = value as Partial<MonitoredJob>;
    if (
      job.id !== jobId ||
      !["queued", "running", "succeeded", "failed", "cancelled"].includes(job.status ?? "") ||
      typeof job.progress !== "number" || !Number.isFinite(job.progress) ||
      typeof job.message !== "string"
    ) return null;
    return job as MonitoredJob;
  } catch {
    return null;
  }
}

/** One polling loop survives stream disconnects and transient network failures. */
export function monitorJob({ load, publish, pollMs = 1000, subscribe, shouldRetry }: {
  load: () => Promise<MonitoredJob>;
  publish: (snapshot: MonitorSnapshot) => void;
  pollMs?: number;
  subscribe?: (receive: (job: MonitoredJob) => void) => () => void;
  shouldRetry?: (cause: unknown) => boolean;
}): () => void {
  let disposed = false;
  let terminal = false;
  let latest: MonitoredJob | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const subscription: { close?: () => void } = {};

  const receive = (job: MonitoredJob) => {
    if (disposed || terminal) return;
    // An older HTTP response must not overwrite a more recent SSE update.
    if (latest && job.updated_at < latest.updated_at) return;
    latest = job;
    terminal = !isActiveJob(job);
    publish({ job, error: null });
    if (terminal) {
      if (timer) clearTimeout(timer);
      subscription.close?.();
    }
  };

  const pull = async () => {
    try {
      receive(await load());
    } catch (cause) {
      if (!disposed && !terminal) {
        if (shouldRetry && !shouldRetry(cause)) {
          terminal = true;
          subscription.close?.();
          publish({ job: latest, error: "This job is no longer available. Reload saved results or start a new run.", stopped: true });
          return;
        }
        publish({
          job: latest,
          error: `Progress connection interrupted. Retrying… ${cause instanceof Error ? cause.message : String(cause)}`,
        });
      }
    }
    if (!disposed && !terminal) timer = setTimeout(() => void pull(), Math.max(100, pollMs));
  };

  subscription.close = subscribe?.(receive);
  if (terminal) subscription.close?.();
  void pull();
  return () => {
    disposed = true;
    if (timer) clearTimeout(timer);
    subscription.close?.();
  };
}

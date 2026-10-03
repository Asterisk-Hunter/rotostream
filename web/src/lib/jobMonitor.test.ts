import assert from "node:assert/strict";
import { test } from "node:test";
import { isActiveJob, monitorJob, parseJobMessage } from "./jobMonitor.ts";
import type { MonitorSnapshot, MonitoredJob } from "./jobMonitor.ts";

const job = (status: MonitoredJob["status"], updated = "2026-10-03T10:00:00Z"): MonitoredJob => ({
  id: "job-1", video_id: "video-1", kind: "track", status,
  progress: status === "succeeded" ? 1 : 0.5, message: status, error: null,
  created_at: updated, updated_at: updated, result: null,
});
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

test("progress messages reject malformed payloads and other jobs", () => {
  assert.equal(parseJobMessage("invalid", "job-1"), null);
  assert.equal(parseJobMessage("{}", "job-1"), null);
  assert.equal(parseJobMessage(JSON.stringify(job("running")), "other"), null);
  assert.equal(parseJobMessage(JSON.stringify({ ...job("running"), status: "unknown" }), "job-1"), null);
  assert.equal(parseJobMessage(JSON.stringify(job("running")), "job-1")?.status, "running");
});

test("all terminal statuses stop the active indicator", () => {
  for (const status of ["failed", "cancelled", "succeeded"] as const) assert.equal(isActiveJob(job(status)), false);
  assert.equal(isActiveJob(job("queued")), true);
  assert.equal(isActiveJob(job("running")), true);
});

test("a first network failure is visible and polling retries to completion", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const snapshots: MonitorSnapshot[] = [];
  let closed = 0;
  const stop = monitorJob({
    pollMs: 100,
    load: async () => { if (++calls === 1) throw new Error("offline"); return job("succeeded"); },
    publish: (snapshot) => snapshots.push(snapshot),
    subscribe: () => () => { closed++; },
  });
  await flush();
  assert.equal(snapshots[0].job, null);
  assert.match(snapshots[0].error ?? "", /offline/);
  context.mock.timers.tick(100);
  await flush();
  assert.equal(snapshots.at(-1)?.job?.status, "succeeded");
  assert.equal(snapshots.at(-1)?.error, null);
  assert.equal(closed, 1);
  context.mock.timers.tick(1000);
  await flush();
  assert.equal(calls, 2);
  stop();
});

test("a terminal stream event wins over an older in-flight poll", async () => {
  const snapshots: MonitorSnapshot[] = [];
  let complete!: (value: MonitoredJob) => void;
  let receive!: (value: MonitoredJob) => void;
  const stop = monitorJob({
    load: () => new Promise((resolve) => { complete = resolve; }),
    publish: (snapshot) => snapshots.push(snapshot),
    subscribe: (callback) => { receive = callback; return () => {}; },
  });
  receive(job("succeeded", "2026-10-03T10:00:02Z"));
  complete(job("running"));
  await flush();
  assert.deepEqual(snapshots.map((snapshot) => snapshot.job?.status), ["succeeded"]);
  stop();
});

test("cleanup discards in-flight updates after clip selection changes", async () => {
  const snapshots: MonitorSnapshot[] = [];
  let complete!: (value: MonitoredJob) => void;
  const stop = monitorJob({
    load: () => new Promise((resolve) => { complete = resolve; }),
    publish: (snapshot) => snapshots.push(snapshot),
  });
  stop();
  complete(job("running"));
  await flush();
  assert.equal(snapshots.length, 0);
});

test("a missing job after server restart stops polling and allows recovery", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const snapshots: MonitorSnapshot[] = [];
  let calls = 0;
  const stop = monitorJob({
    load: async () => { calls++; throw new Error("missing"); },
    publish: (snapshot) => snapshots.push(snapshot),
    shouldRetry: () => false,
  });
  await flush();
  assert.equal(snapshots[0].stopped, true);
  assert.match(snapshots[0].error ?? "", /no longer available/);
  context.mock.timers.tick(10000);
  await flush();
  assert.equal(calls, 1);
  stop();
});

test("stream updates cannot move progress back to an older snapshot", async () => {
  const snapshots: MonitorSnapshot[] = [];
  let receive!: (value: MonitoredJob) => void;
  const stop = monitorJob({
    load: async () => job("running"),
    publish: (snapshot) => snapshots.push(snapshot),
    subscribe: (callback) => { receive = callback; return () => {}; },
  });
  receive(job("running", "2026-10-03T10:00:02Z"));
  await flush();
  assert.equal(snapshots.length, 1);
  stop();
});

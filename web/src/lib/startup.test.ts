import assert from "node:assert/strict";
import { test } from "node:test";
import { connectStartup, startupErrorMessage } from "./startup.ts";

const flush = () => new Promise<void>((resolve) => setImmediate(resolve));
const httpError = (status: number) => Object.assign(new Error(`HTTP ${status}`), { status });

test("a cold-start timeout retries after a delay and publishes the recovered result once", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const values: string[] = [];
  const errors: unknown[] = [];
  const stop = connectStartup({
    load: async () => {
      if (++calls === 1) throw new DOMException("signal timed out", "TimeoutError");
      return "ready";
    },
    onReady: (value) => values.push(value),
    onError: (cause) => errors.push(cause),
  });
  await flush();
  context.mock.timers.tick(999);
  await flush();
  assert.equal(calls, 1);
  assert.deepEqual(values, []);
  assert.deepEqual(errors, []);
  context.mock.timers.tick(1);
  await flush();
  assert.deepEqual(values, ["ready"]);
  context.mock.timers.tick(60_000);
  await flush();
  assert.equal(calls, 2);
  stop();
});

test("network failures and gateway unavailable responses can recover", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  for (const cause of [new TypeError("Failed to fetch"), new DOMException("Network request failed", "NetworkError"), httpError(0), httpError(502), httpError(503), httpError(504)]) {
    let calls = 0;
    let result: string | null = null;
    const errors: unknown[] = [];
    const stop = connectStartup({
      load: async () => { if (++calls === 1) throw cause; return "connected"; },
      onReady: (value) => { result = value; },
      onError: (error) => errors.push(error),
    });
    await flush();
    context.mock.timers.tick(1000);
    await flush();
    assert.equal(result, "connected");
    assert.deepEqual(errors, []);
    stop();
  }
});

test("permanent API and application errors fail immediately without an automatic retry", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  for (const cause of [httpError(400), httpError(401), httpError(404), httpError(500), new Error("Invalid response"), new TypeError("Cannot read properties of undefined"), new DOMException("cancelled", "AbortError")]) {
    let calls = 0;
    const errors: unknown[] = [];
    const stop = connectStartup({
      load: async () => { calls++; throw cause; },
      onReady: () => assert.fail("failed requests must not publish data"),
      onError: (error) => errors.push(error),
    });
    await flush();
    context.mock.timers.tick(60_000);
    await flush();
    assert.equal(calls, 1);
    assert.deepEqual(errors, [cause]);
    stop();
  }
});

test("repeated startup failures exhaust a finite retry budget and publish the last error", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const errors: unknown[] = [];
  const last = httpError(503);
  const stop = connectStartup({
    load: async () => { calls++; throw last; },
    onReady: () => assert.fail("failed requests must not publish data"),
    onError: (error) => errors.push(error),
  });
  await flush();
  for (const delay of [1000, 2000, 4000]) {
    context.mock.timers.tick(delay);
    await flush();
  }
  assert.equal(calls, 4);
  assert.deepEqual(errors, [last]);
  context.mock.timers.tick(60_000);
  await flush();
  assert.equal(calls, 4);
  stop();
});

test("cleanup cancels pending retries", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const stop = connectStartup({
    load: async () => { calls++; throw httpError(503); },
    onReady: () => assert.fail("cancelled startup must not publish data"),
    onError: () => assert.fail("cancelled startup must not publish errors"),
  });
  await flush();
  stop();
  context.mock.timers.tick(60_000);
  await flush();
  assert.equal(calls, 1);
});

test("cleanup discards late success and failure from in-flight startup requests", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  for (const fails of [false, true]) {
    let settle!: () => void;
    let calls = 0;
    const stop = connectStartup({
      load: () => {
        calls++;
        return new Promise<string>((resolve, reject) => {
          settle = () => fails ? reject(httpError(503)) : resolve("stale");
        });
      },
      onReady: () => assert.fail("disposed startup must not publish late data"),
      onError: () => assert.fail("disposed startup must not publish late errors"),
    });
    stop();
    settle();
    await flush();
    context.mock.timers.tick(60_000);
    await flush();
    assert.equal(calls, 1);
  }
});

test("timeout and connection errors explain recovery without depending on an API base URL", () => {
  assert.match(startupErrorMessage(new DOMException("signal timed out", "TimeoutError")), /processing service.*too long.*[Rr]etry/);
  assert.match(startupErrorMessage(new TypeError("Failed to fetch")), /connection.*[Rr]etry/);
  assert.match(startupErrorMessage(httpError(503)), /processing service.*[Rr]etry/);
  assert.match(startupErrorMessage(new Error("Access denied")), /Access denied/);
});

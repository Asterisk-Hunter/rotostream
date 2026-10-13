import assert from "node:assert/strict";
import test from "node:test";
import { DemoLimiter, mutationAllowed } from "./demoLimits.ts";

test("upload quota rejects the third upload and recovers after an hour", () => {
  const limiter = new DemoLimiter();
  assert.equal(limiter.check("visitor", "POST", "/api/videos", 0), 0);
  assert.equal(limiter.check("visitor", "POST", "/api/videos", 0), 0);
  assert.equal(limiter.check("visitor", "POST", "/api/videos", 0), 3600);
  assert.equal(limiter.check("visitor", "POST", "/api/videos", 3_600_000), 0);
});

test("processing quota includes previews, tracking and exports without throttling SSE polling", () => {
  const limiter = new DemoLimiter();
  for (let count = 0; count < 20; count++) {
    assert.equal(limiter.check("visitor", "POST", "/api/videos/clip/preview", 0), 0);
  }
  assert.equal(limiter.check("visitor", "POST", "/api/videos/clip/track", 0), 3600);
  assert.equal(limiter.check("visitor", "POST", "/api/videos/clip/exports", 0), 3600);
  assert.equal(limiter.check("visitor", "GET", "/api/jobs/job/events", 0), 0);
});

test("shared quota bounds uploads from different visitors in the same instance", () => {
  const limiter = new DemoLimiter();
  for (let count = 0; count < 20; count++) {
    assert.equal(limiter.check(`visitor-${count}`, "POST", "/api/videos", 0), 0);
  }
  assert.equal(limiter.check("another-visitor", "POST", "/api/videos", 0), 3600);
});

test("mutations require same-origin browser requests and public deletes are blocked", () => {
  assert.equal(mutationAllowed("POST", "https://demo.test", "https://demo.test", false), true);
  assert.equal(mutationAllowed("POST", null, "https://demo.test", false), false);
  assert.equal(mutationAllowed("POST", "https://other.test", "https://demo.test", false), false);
  assert.equal(mutationAllowed("DELETE", "https://demo.test", "https://demo.test", false), false);
  assert.equal(mutationAllowed("DELETE", "https://demo.test", "https://demo.test", true), true);
  assert.equal(mutationAllowed("GET", null, "https://demo.test", false), true);
});

test("read allowance supports playback and rejects request floods until the next minute", () => {
  const limiter = new DemoLimiter();
  for (let count = 0; count < 3000; count++) {
    assert.equal(limiter.check("viewer", "GET", "/api/videos/clip/frames/1", 0), 0);
  }
  assert.equal(limiter.check("viewer", "GET", "/api/videos/clip/frames/1", 0), 60);
  assert.equal(limiter.check("viewer", "GET", "/api/videos/clip/frames/1", 60_000), 0);
});

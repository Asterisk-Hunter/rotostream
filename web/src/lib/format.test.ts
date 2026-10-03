import assert from "node:assert/strict";
import { test } from "node:test";
import { formatBytes, formatDuration, formatScore, formatTimecode } from "./format.ts";

test("duration rounding carries seconds into minutes", () => {
  assert.equal(formatDuration(119.9), "2m 00s");
  assert.equal(formatDuration(60), "1m 00s");
});
test("timecode rounding never shows a 60-second component", () => {
  assert.equal(formatTimecode(599999, 10000), "1:00.00");
  assert.equal(formatTimecode(30, 30), "0:01.00");
});
test("invalid metrics and metadata have useful display fallbacks", () => {
  assert.equal(formatBytes(Number.NaN), "0 B");
  assert.equal(formatDuration(Number.POSITIVE_INFINITY), "0s");
  assert.equal(formatScore(Number.NaN), "—");
  assert.equal(formatTimecode(4, 0), "#4");
});

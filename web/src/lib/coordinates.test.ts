import assert from "node:assert/strict";
import { test } from "node:test";
import { frameCoordinates } from "./coordinates.ts";

const rect = { left: 10, top: 20, width: 400, height: 200 };
test("rendered pointer coordinates map to source frame pixels", () => {
  assert.deepEqual(frameCoordinates(210, 120, rect, 960, 480), { x: 480, y: 240 });
});
test("frame edge and out-of-bounds pointer events stay inside the mask", () => {
  assert.deepEqual(frameCoordinates(410, 220, rect, 960, 480), { x: 959, y: 479 });
  assert.deepEqual(frameCoordinates(-50, -50, rect, 960, 480), { x: 0, y: 0 });
});
test("unmeasured surfaces cannot create invalid prompts", () => {
  assert.equal(frameCoordinates(10, 20, { ...rect, width: 0 }, 960, 480), null);
  assert.equal(frameCoordinates(10, 20, rect, 0, 480), null);
});

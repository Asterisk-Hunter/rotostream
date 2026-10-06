import assert from "node:assert/strict";
import test from "node:test";
import { isPublicGuide } from "./routeAccess.ts";

test("the workflow guide is public so new editors can learn before signing in", () => {
  assert.equal(isPublicGuide("/docs"), true);
  assert.equal(isPublicGuide("/docs/private"), false);
  assert.equal(isPublicGuide("/"), false);
});

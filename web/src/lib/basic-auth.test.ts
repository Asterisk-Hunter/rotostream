import assert from "node:assert/strict";
import test from "node:test";
import { matchesBasicCredentials, parseBasicCredentials } from "./basic-auth.ts";

const validHeader = `Basic ${btoa("editor:secret-value")}`;

test("parses a Basic authorization header", () => {
  assert.deepEqual(parseBasicCredentials(validHeader), {
    username: "editor",
    password: "secret-value",
  });
});

test("rejects missing, malformed, and incomplete credentials", () => {
  assert.equal(parseBasicCredentials(null), null);
  assert.equal(parseBasicCredentials("Bearer token"), null);
  assert.equal(parseBasicCredentials("Basic !invalid!"), null);
  assert.equal(parseBasicCredentials(`Basic ${btoa("missing-separator")}`), null);
});

test("matches credentials exactly", () => {
  assert.equal(matchesBasicCredentials(validHeader, "editor", "secret-value"), true);
  assert.equal(matchesBasicCredentials(validHeader, "Editor", "secret-value"), false);
  assert.equal(matchesBasicCredentials(validHeader, "editor", "Secret-value"), false);
});

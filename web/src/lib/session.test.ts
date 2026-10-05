import assert from "node:assert/strict";
import test from "node:test";
import { createSessionToken, getSessionSigningSecret, SESSION_TTL_SECONDS, verifySessionToken } from "./session.ts";

const secret = "test-session-secret-that-is-long-enough";
const now = 1_800_000_000;

test("session signing key is derived from the existing editor password", () => {
  assert.equal(getSessionSigningSecret(secret), getSessionSigningSecret(secret));
  assert.notEqual(getSessionSigningSecret(secret), getSessionSigningSecret("another-password"));
  assert.equal(getSessionSigningSecret(undefined), null);
});

test("session token verifies for its intended editor until expiry", () => {
  const token = createSessionToken("editor", secret, now);
  assert.equal(verifySessionToken(token, secret, "editor", now), true);
  assert.equal(verifySessionToken(token, secret, "another-editor", now), false);
  assert.equal(verifySessionToken(token, secret, "editor", now + SESSION_TTL_SECONDS), false);
});

test("session token rejects tampering and malformed input", () => {
  const token = createSessionToken("editor", secret, now);
  assert.equal(verifySessionToken(`${token}x`, secret, "editor", now), false);
  assert.equal(verifySessionToken(token, "different-secret", "editor", now), false);
  assert.equal(verifySessionToken("not-a-token", secret, "editor", now), false);
  assert.equal(verifySessionToken(undefined, secret, "editor", now), false);
});

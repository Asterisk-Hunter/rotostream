import assert from "node:assert/strict";
import test from "node:test";

import {
  frameState,
  nextProblemFrame,
  promptSignature,
  readiness,
  reviewSummary,
  summarizeScores,
} from "./review.ts";
import type { FrameScore, Prompt, Session } from "./types.ts";

function score(frame: number, value: number, present = true, prompted = false): FrameScore {
  return { frame_index: frame, score: value, object_present: present, prompted };
}

function promptAt(frame: number, positive = true): Prompt {
  return {
    frame_index: frame,
    points: [{ x: 10, y: 10, positive }],
    box: null,
  };
}

test("a run with a missing mask is never reported as sound", () => {
  const scores = [score(0, 0.9, true, true), score(1, 0.0, false), score(2, 0.8)];
  const summary = summarizeScores(scores, 3, [promptAt(0)]);
  assert.deepEqual(summary.absentFrames, [1]);
  assert.equal(summary.sound, false);
  assert.equal(readiness(summary)?.tone, "error");
  assert.match(readiness(summary)!.headline, /No mask on 1 of 3 frames/);
});

test("a background-only prompt frame is the user's instruction, not lost tracking", () => {
  const prompts = [promptAt(0), promptAt(1, false)];
  const scores = [score(0, 0.9, true, true), score(1, 0.0, false, true), score(2, 0.85)];
  const summary = summarizeScores(scores, 3, prompts);
  assert.deepEqual(summary.absentFrames, []);
  assert.deepEqual(summary.backgroundOnlyFrames, [1]);
  assert.equal(summary.sound, true);
});

test("low-confidence frames are surfaced and warned about", () => {
  const scores = [
    score(0, 0.9, true, true),
    ...Array.from({ length: 4 }, (_, index) => score(index + 1, 0.2)),
    ...Array.from({ length: 35 }, (_, index) => score(index + 5, 0.95)),
  ];
  const summary = summarizeScores(scores, 40, [promptAt(0)]);
  assert.equal(summary.lowConfidenceFrames.length, 4);
  assert.equal(summary.sound, false);
  assert.equal(readiness(summary)?.tone, "warn");
});

test("a prompted frame is never called low confidence", () => {
  const summary = summarizeScores([score(0, 0.2, true, true)], 1, [promptAt(0)]);
  assert.deepEqual(summary.lowConfidenceFrames, []);
  assert.equal(summary.sound, true);
});

test("a sound run states what was measured instead of declaring success", () => {
  const scores = Array.from({ length: 10 }, (_, index) => score(index, 0.93));
  const summary = summarizeScores(scores, 10, [promptAt(0)]);
  const verdict = readiness(summary)!;
  assert.equal(verdict.tone, "ok");
  assert.match(verdict.headline, /Mask on 10 of 10 frames/);
  assert.doesNotMatch(verdict.headline.toLowerCase(), /ready/);
  assert.match(verdict.detail, /export/);
});

test("nothing tracked means nothing claimed", () => {
  assert.equal(readiness(null), null);
  assert.equal(reviewSummary(null), null);
});

test("the API summary wins when it is present", () => {
  const session = {
    id: "s1",
    video_id: "v",
    model: "naive",
    created_at: "",
    n_tracked: 4,
    n_frames: 4,
    prompt_frames: [0],
    mean_score: 0.9,
    absent_frames: [2],
    elapsed_s: 1,
    cancelled: false,
    status: "succeeded",
    error: null,
    scores: [score(0, 0.9, true, true)],
    memory: {},
    quality: {
      threshold: 0.5,
      n_frames: 4,
      n_masked: 3,
      coverage: 0.75,
      absent_frames: [2],
      low_confidence_frames: [3],
      background_only_frames: [],
      problem_frames: [2, 3],
      mean_score: 0.88,
      min_score: 0.4,
      sound: false,
    },
  } as unknown as Session;
  const summary = reviewSummary(session)!;
  assert.deepEqual(summary.problemFrames, [2, 3]);
  assert.equal(summary.nMasked, 3);
  assert.equal(summary.sound, false);
});

test("problem navigation seeks forward, backward and wraps", () => {
  assert.equal(nextProblemFrame([4, 9, 20], 0, 1), 4);
  assert.equal(nextProblemFrame([4, 9, 20], 4, 1), 9);
  assert.equal(nextProblemFrame([4, 9, 20], 9, -1), 4);
  assert.equal(nextProblemFrame([4, 9, 20], 0, -1), 20);
  assert.equal(nextProblemFrame([], 3, 1), null);
});

test("frame states separate absent, low, prompt, background and untracked frames", () => {
  const summary = summarizeScores(
    [score(0, 0.9, true, true), score(1, 0.2), score(2, 0.0, false)],
    4,
    [promptAt(0), promptAt(3, false)],
  );
  assert.equal(frameState(undefined, summary, [0], 5), "untracked");
  assert.equal(frameState(score(1, 0.2), summary, [0], 1), "low");
  assert.equal(frameState(score(2, 0.0, false), summary, [0], 2), "absent");
  assert.equal(frameState(score(3, 0.0, false, true), summary, [0], 3), "background");
  assert.equal(frameState(score(0, 0.9, true, true), summary, [0], 0), "prompted");
});

test("prompt signatures detect a changed prompt set", () => {
  const base = [promptAt(0), promptAt(4)];
  assert.equal(promptSignature(base), promptSignature([promptAt(4), promptAt(0)]));
  assert.notEqual(promptSignature(base), promptSignature([promptAt(0), promptAt(5)]));
  assert.notEqual(promptSignature(base), promptSignature([promptAt(0)]));
  assert.equal(promptSignature([]), "");
  assert.equal(promptSignature(undefined), "");
});

/**
 * Review rules for a tracking run.
 *
 * The studio must never imply a mask is finished when parts of the clip are
 * missing or weak, so the thresholds and the wording live in one place here and
 * are covered by tests. Everything is derived from the scores the API returns, so
 * the panel, the timeline and the export warning all agree.
 */
import type { FrameScore, Prompt, Session } from "./types";

/** Frames at or below this score need a human look; matches the API default. */
export const LOW_CONFIDENCE = 0.5;
/** The share of low-confidence frames that still counts as a sound run. */
export const SOUND_LOW_CONFIDENCE_SHARE = 0.05;

export type FrameState = "untracked" | "absent" | "background" | "low" | "prompted" | "ok";

export interface ReviewSummary {
  nFrames: number;
  nMasked: number;
  coverage: number;
  absentFrames: number[];
  lowConfidenceFrames: number[];
  backgroundOnlyFrames: number[];
  problemFrames: number[];
  meanScore: number;
  minScore: number;
  threshold: number;
  sound: boolean;
}

function hasPositiveSeed(prompt: Prompt): boolean {
  return Boolean(prompt.box) || prompt.points.some((point) => point.positive);
}

/**
 * Derive the review summary from raw scores.
 *
 * Used when the API did not send its own summary (older sessions, or a job result
 * mid-flight). The rules mirror `api/app/pipeline.py::summarize_quality`:
 *
 * - a propagated frame with no object is *absent*;
 * - a frame the user marked as background on purpose is neither absent nor lost;
 * - a low score on a frame the user prompted says nothing about the mask quality.
 */
export function summarizeScores(
  scores: FrameScore[],
  nFrames: number,
  prompts: Prompt[] = [],
): ReviewSummary {
  const prompted = new Set(prompts.map((prompt) => prompt.frame_index));
  const backgroundOnly = new Set(
    prompts.filter((prompt) => !hasPositiveSeed(prompt)).map((prompt) => prompt.frame_index),
  );

  const absent: number[] = [];
  const low: number[] = [];
  const masked: number[] = [];
  for (const record of scores) {
    if (record.frame_index >= nFrames) continue;
    if (record.prompted) prompted.add(record.frame_index);
    const deliberateBackground = backgroundOnly.has(record.frame_index) || (record.prompted && !record.object_present);
    if (!record.object_present) {
      if (!deliberateBackground && record.frame_index < nFrames) absent.push(record.frame_index);
      continue;
    }
    masked.push(record.frame_index);
    if (record.score < LOW_CONFIDENCE && !prompted.has(record.frame_index)) low.push(record.frame_index);
  }

  const values = scores.filter((record) => record.object_present).map((record) => record.score);
  const problem = Array.from(new Set([...absent, ...low])).sort((a, b) => a - b);
  const coverage = nFrames > 0 ? masked.length / nFrames : 0;
  const allowance = Math.max(0, Math.floor(SOUND_LOW_CONFIDENCE_SHARE * Math.max(1, nFrames)));

  return {
    nFrames,
    nMasked: masked.length,
    coverage,
    absentFrames: absent.sort((a, b) => a - b),
    lowConfidenceFrames: low.sort((a, b) => a - b),
    backgroundOnlyFrames: Array.from(backgroundOnly).sort((a, b) => a - b),
    problemFrames: problem,
    meanScore: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0,
    minScore: values.length ? Math.min(...values) : 0,
    threshold: LOW_CONFIDENCE,
    sound: absent.length === 0 && low.length <= allowance,
  };
}

/** The API's summary when it is present, otherwise a local derivation. */
export function reviewSummary(session: Session | null | undefined): ReviewSummary | null {
  if (!session) return null;
  const nFrames = session.n_frames || session.scores.length;
  const derived = summarizeScores(session.scores ?? [], nFrames, session.prompts ?? []);
  const quality = session.quality;
  if (!quality || !quality.n_frames) return derived;
  return {
    nFrames: quality.n_frames,
    nMasked: quality.n_masked,
    coverage: quality.coverage,
    absentFrames: quality.absent_frames ?? derived.absentFrames,
    lowConfidenceFrames: quality.low_confidence_frames ?? derived.lowConfidenceFrames,
    backgroundOnlyFrames: quality.background_only_frames ?? derived.backgroundOnlyFrames,
    problemFrames: quality.problem_frames ?? derived.problemFrames,
    meanScore: quality.mean_score,
    minScore: quality.min_score,
    threshold: quality.threshold || LOW_CONFIDENCE,
    sound: quality.sound,
  };
}

/** How one frame should be drawn and described in the timeline. */
export function frameState(
  record: FrameScore | undefined,
  summary: ReviewSummary | null,
  promptFrames: number[],
  index: number,
): FrameState {
  if (summary?.backgroundOnlyFrames.includes(index)) return "background";
  if (promptFrames.includes(index)) return "prompted";
  if (!record) return "untracked";
  if (!record.object_present) return "absent";
  if (record.score < (summary?.threshold ?? LOW_CONFIDENCE)) return "low";
  return "ok";
}

/** Build a constant-time frame classifier for timeline-wide rendering. */
export function createFrameStateLookup(
  summary: ReviewSummary | null,
  promptFrames: number[],
): (record: FrameScore | undefined, index: number) => FrameState {
  const backgroundFrames = new Set(summary?.backgroundOnlyFrames ?? []);
  const promptedFrames = new Set(promptFrames);
  const threshold = summary?.threshold ?? LOW_CONFIDENCE;
  return (record, index) => {
    if (backgroundFrames.has(index)) return "background";
    if (promptedFrames.has(index)) return "prompted";
    if (!record) return "untracked";
    if (!record.object_present) return "absent";
    if (record.score < threshold) return "low";
    return "ok";
  };
}

/** Nearest frame that needs attention, in the given direction. Wraps once. */
export function nextProblemFrame(
  problemFrames: number[],
  index: number,
  direction: 1 | -1,
): number | null {
  if (problemFrames.length === 0) return null;
  const ordered = [...problemFrames].sort((a, b) => a - b);
  if (direction === 1) {
    return ordered.find((frame) => frame > index) ?? ordered[0];
  }
  const before = ordered.filter((frame) => frame < index);
  return before.length ? before[before.length - 1] : ordered[ordered.length - 1];
}

/** Stable identity of a prompt set, for "changed since this run" comparisons. */
export function promptSignature(prompts: Prompt[] | undefined): string {
  if (!prompts || prompts.length === 0) return "";
  return [...prompts]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map((prompt) => {
      const points = [...prompt.points]
        .map((point) => `${point.positive ? "+" : "-"}${Math.round(point.x)},${Math.round(point.y)}`)
        .sort()
        .join(";");
      const box = prompt.box
        ? `b${Math.round(prompt.box.x0)},${Math.round(prompt.box.y0)},${Math.round(prompt.box.x1)},${Math.round(prompt.box.y1)}`
        : "";
      return `${prompt.frame_index}|${points}|${box}`;
    })
    .join(" ");
}

export interface Readiness {
  tone: "ok" | "warn" | "error";
  headline: string;
  detail: string;
}

/**
 * What the studio is allowed to claim about a finished run.
 *
 * There is deliberately no "ready" state: a sound run still says what was measured
 * and what to check, and a run with missing or weak frames says plainly that the
 * clip is not finished and what to do next.
 */
export function readiness(summary: ReviewSummary | null): Readiness | null {
  if (!summary || summary.nFrames === 0) return null;
  const absent = summary.absentFrames.length;
  const low = summary.lowConfidenceFrames.length;
  const percent = Math.round(summary.coverage * 100);

  if (absent > 0) {
    return {
      tone: "error",
      headline: `No mask on ${absent} of ${summary.nFrames} frames.`,
      detail:
        "Tracking lost the object there. Open those frames, add a foreground click or a box on the object, then track again. Exporting now renders them with the background only.",
    };
  }
  if (!summary.sound) {
    return {
      tone: "warn",
      headline: `${low} of ${summary.nFrames} frames are low confidence (below ${summary.threshold.toFixed(2)}).`,
      detail:
        "The mask is present but the tracker is unsure. Check those frames, correct them with clicks, and track again before exporting.",
    };
  }
  return {
    tone: "ok",
    headline: `Mask on ${summary.nMasked} of ${summary.nFrames} frames (${percent}%), lowest score ${summary.minScore.toFixed(2)}.`,
    detail: "Step through the timeline to confirm the outline, then export.",
  };
}

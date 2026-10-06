"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { assetUrl } from "@/lib/api";
import { formatScore } from "@/lib/format";
import type { ReviewSummary } from "@/lib/review";
import { createFrameStateLookup } from "@/lib/review";
import type { FrameScore } from "@/lib/types";

import { Button, cx } from "./ui";

interface Props {
  videoId: string;
  nFrames: number;
  index: number;
  scores: FrameScore[];
  promptFrames: number[];
  summary: ReviewSummary | null;
  fps: number;
  onIndexChange: (index: number) => void;
  /** Jump to the previous/next frame that needs attention. */
  onJumpProblem: (direction: 1 | -1) => void;
}

const STATE_COLOUR = {
  untracked: "var(--color-ink-750)",
  absent: "var(--color-negative)",
  background: "var(--color-ink-600)",
  low: "var(--color-warn)",
  prompted: "var(--color-accent-400)",
  ok: "var(--color-positive)",
} as const;

const STATE_LABEL = {
  untracked: "not tracked",
  absent: "no mask",
  background: "marked background",
  low: "low confidence",
  prompted: "prompted",
  ok: "tracked",
} as const;

const REVIEW_PREVIEW_HZ = 8;

/**
 * One bar per frame: colour is the review state, height is the tracker's own
 * confidence. Frames that are missing or weak are marked rather than averaged away,
 * and there is a control to step straight to them.
 */
export function Timeline({
  videoId,
  nFrames,
  index,
  scores,
  promptFrames,
  summary,
  fps,
  onIndexChange,
  onJumpProblem,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const indexRef = useRef(index);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);
  const last = Math.max(0, nFrames - 1);
  const byFrame = useMemo(
    () => new Map(scores.map((score) => [score.frame_index, score])),
    [scores],
  );
  const classifyFrame = useMemo(
    () => createFrameStateLookup(summary, promptFrames),
    [summary, promptFrames],
  );

  const seekFromEvent = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const fraction = (event.clientX - rect.left) / rect.width;
    onIndexChange(Math.min(last, Math.max(0, Math.round(fraction * last))));
  };

  const current = byFrame.get(index);
  const problems = summary?.problemFrames ?? [];
  const problemState = classifyFrame(current, index);
  const sampleCount = Math.min(160, Math.max(1, nFrames));
  const bins = useMemo(() => {
    const result = Array.from({ length: sampleCount }, () => ({
      total: 0, present: 0, score: 0, absent: false, low: false, prompt: false, background: false,
    }));
    const binOf = (frame: number) => Math.min(sampleCount - 1, Math.floor((frame * sampleCount) / Math.max(1, nFrames)));
    for (const record of scores) {
      const bin = result[binOf(record.frame_index)];
      if (!bin) continue;
      bin.total++;
      const state = classifyFrame(record, record.frame_index);
      if (state === "absent") bin.absent = true;
      else if (state === "background") bin.background = true;
      else if (state === "low") bin.low = true;
      if (state === "prompted") bin.prompt = true;
      if (record.object_present) {
        bin.present++;
        bin.score += record.score;
      }
    }
    return result;
  }, [sampleCount, nFrames, scores, classifyFrame]);

  const thumbnails = useMemo(() =>
    Array.from({ length: Math.min(10, nFrames) }, (_, slot) =>
      Math.round((slot * last) / Math.max(1, Math.min(10, nFrames) - 1)),
    ), [nFrames, last]);

  useEffect(() => {
    if (!playing) return;
    const startedAt = performance.now();
    const startIndex = indexRef.current;
    const timer = setInterval(() => {
      const elapsedFrames = Math.floor(((performance.now() - startedAt) * Math.max(1, fps)) / 1000);
      const nextIndex = startIndex + elapsedFrames;
      if (nextIndex >= last) {
        onIndexChange(last);
        setPlaying(false);
      } else {
        onIndexChange(nextIndex);
      }
    }, 1000 / REVIEW_PREVIEW_HZ);
    return () => clearInterval(timer);
  }, [playing, last, fps, onIndexChange]);

  const absentCount = summary?.absentFrames.length ?? 0;
  const lowCount = summary?.lowConfidenceFrames.length ?? 0;
  const tone = absentCount > 0 ? "text-negative" : lowCount > 0 ? "text-warn" : "text-ink-300";

  return (
    <div className="timeline-shell shrink-0 rounded-panel border px-3.5 py-3">
      <div className="timeline-filmstrip mb-2" aria-label="Clip frame overview">
        {thumbnails.map((frame) => (
          <button
            key={frame}
            type="button"
            onClick={() => onIndexChange(frame)}
            aria-current={frame === index ? "true" : "false"}
            aria-label={`Go to frame ${frame + 1}`}
            title={`Frame ${frame + 1}`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={assetUrl.frame(videoId, frame)} alt="" loading="lazy" />
          </button>
        ))}
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${sampleCount} 1`}
          preserveAspectRatio="none"
          onClick={seekFromEvent}
          className="h-12 w-full cursor-pointer rounded-[6px] border border-ink-800 bg-ink-950/50"
          role="img"
          aria-label={
            summary
              ? `Per-frame mask quality: ${absentCount} frames without a mask, ${lowCount} low confidence`
              : "Per-frame mask quality"
          }
        >
          <line
            x1={0}
            y1={0.5}
            x2={sampleCount}
            y2={0.5}
            stroke="var(--color-ink-700)"
            strokeWidth={0.005}
          />
          {bins.map((sample, bin) => {
            const tracked = sample.total > 0;
            const confidence = sample.present ? sample.score / sample.present : 0;
            const state = !tracked
              ? "untracked"
              : sample.absent
                ? "absent"
                : sample.background
                  ? "background"
                  : sample.low
                    ? "low"
                    : sample.prompt
                      ? "prompted"
                      : "ok";
            // Missing frames get a full-height mark: "no mask here" must be as loud
            // as a confident frame, otherwise a lost object reads as a quiet gap.
            const height = state === "absent" ? 1 : tracked ? Math.max(0.06, confidence) : 0.03;
            return (
              <rect
                key={bin}
                x={bin + 0.1}
                y={1 - height}
                width={0.8}
                height={height}
                fill={STATE_COLOUR[state]}
                opacity={state === "untracked" ? 0.35 : 0.9}
              />
            );
          })}
        </svg>

        <div
          className="pointer-events-none absolute inset-y-0 w-px bg-accent-300"
          style={{ left: `${(index / Math.max(1, last)) * 100}%` }}
        >
          <span className="absolute -top-1 left-1/2 h-1.5 w-1.5 -translate-x-1/2 rotate-45 bg-accent-300" />
        </div>
      </div>

      <input
        type="range"
        min={0}
        max={last}
        value={index}
        aria-label="Frame"
        onChange={(event) => onIndexChange(Number(event.target.value))}
        className="range-track mt-2 w-full cursor-pointer accent-[var(--color-accent-400)]"
      />

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button
          variant="secondary"
          className="px-2.5 py-1.5 text-[11px]"
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? "Pause review preview" : "Play sampled review preview"}
          aria-pressed={playing}
          disabled={nFrames <= 1}
        >
          {playing ? "Pause" : "Play preview"}
        </Button>
        <Button
          variant="ghost"
          className="px-2.5 py-1.5 text-[11px]"
          onClick={() => onJumpProblem(-1)}
          disabled={problems.length === 0}
          aria-label="Previous frame that needs review"
        >
          Problem ←
        </Button>
        <Button
          variant="ghost"
          className="px-2.5 py-1.5 text-[11px]"
          onClick={() => onJumpProblem(1)}
          disabled={problems.length === 0}
          aria-label="Next frame that needs review"
        >
          Problem →
        </Button>
        <p className={cx("ml-auto text-[11px]", tone)} role="status">
          {summary
            ? `Mask on ${summary.nMasked}/${summary.nFrames} frames` +
              (absentCount ? ` · ${absentCount} without a mask` : "") +
              (lowCount ? ` · ${lowCount} low confidence` : "") +
              (problems.length === 0 ? " · nothing to fix" : "")
            : "Not tracked yet"}
        </p>
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-500">
        <span>Preview samples at up to {REVIEW_PREVIEW_HZ} frames/s</span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-[2px] bg-accent-400" /> prompted
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-[2px] bg-positive" /> tracked
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-[2px] bg-warn" /> low confidence
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-[2px] bg-negative" /> no mask
        </span>
        <span className="ml-auto font-mono">
          {current
            ? `${STATE_LABEL[problemState]}${current.object_present ? ` · score ${formatScore(current.score)}` : ""}`
            : "no data for this frame"}
        </span>
      </div>
    </div>
  );
}

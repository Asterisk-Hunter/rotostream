"use client";

import type { MouseEvent } from "react";

import { formatScore } from "@/lib/format";
import type { FrameScore } from "@/lib/types";

import { Badge } from "./ui";

interface Props {
  nFrames: number;
  index: number;
  scores: FrameScore[];
  promptFrames: number[];
  onIndexChange: (index: number) => void;
}

/**
 * Each bar is one frame: height is the tracker's confidence, colour separates
 * propagated from prompted from reported-absent. Clicking the strip seeks.
 */
export function Timeline({ nFrames, index, scores, promptFrames, onIndexChange }: Props) {
  const byFrame = new Map(scores.map((score) => [score.frame_index, score]));
  const prompted = new Set(promptFrames);
  const last = Math.max(0, nFrames - 1);

  const seekFromEvent = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const fraction = (event.clientX - rect.left) / rect.width;
    onIndexChange(Math.min(last, Math.max(0, Math.round(fraction * last))));
  };

  const current = byFrame.get(index);

  return (
    <div className="rounded-panel border border-ink-800 bg-ink-900/55 px-3.5 py-3">
      <div className="relative">
        <svg
          viewBox={`0 0 ${Math.max(nFrames, 1)} 1`}
          preserveAspectRatio="none"
          onClick={seekFromEvent}
          className="h-12 w-full cursor-pointer rounded-lg border border-ink-800 bg-ink-950/50"
          role="img"
          aria-label="Per-frame mask confidence"
        >
          <line
            x1={0}
            y1={0.5}
            x2={nFrames}
            y2={0.5}
            stroke="var(--color-ink-700)"
            strokeWidth={0.005}
          />
          {Array.from({ length: nFrames }, (_, frame) => {
            const record = byFrame.get(frame);
            const confidence = record ? record.score : 0;
            const height = record ? Math.max(0.04, confidence) : 0.02;
            const colour = !record
              ? "var(--color-ink-750)"
              : !record.object_present
                ? "var(--color-negative)"
                : prompted.has(frame)
                  ? "var(--color-accent-400)"
                  : "var(--color-positive)";
            return (
              <rect
                key={frame}
                x={frame + 0.15}
                y={1 - height}
                width={0.7}
                height={height}
                fill={colour}
                opacity={record ? 0.85 : 0.4}
              />
            );
          })}
        </svg>

        {/* Playhead */}
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

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[10px] text-ink-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-accent-400" /> prompted
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-positive" /> propagated
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-negative" /> object absent
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          {current ? (
            <>
              <Badge tone={current.object_present ? "positive" : "negative"}>
                {current.object_present ? "present" : "absent"}
              </Badge>
              <span className="tnum font-mono">score {formatScore(current.score)}</span>
            </>
          ) : (
            <span className="font-mono">not tracked yet</span>
          )}
        </span>
      </div>
    </div>
  );
}

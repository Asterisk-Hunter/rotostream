"use client";

import { useEffect, useState, type MouseEvent } from "react";

import { formatScore } from "@/lib/format";
import { assetUrl } from "@/lib/api";
import type { FrameScore } from "@/lib/types";

import { Button } from "./ui";

interface Props {
  videoId: string;
  nFrames: number;
  index: number;
  scores: FrameScore[];
  promptFrames: number[];
  fps: number;
  onIndexChange: (index: number) => void;
}

/**
 * Each bar is one frame: height is the tracker's confidence, colour separates
 * propagated from prompted from reported-absent. Clicking the strip seeks.
 */
export function Timeline({ videoId, nFrames, index, scores, promptFrames, fps, onIndexChange }: Props) {
  const [playing, setPlaying] = useState(false);
  const byFrame = new Map(scores.map((score) => [score.frame_index, score]));
  const last = Math.max(0, nFrames - 1);

  const seekFromEvent = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const fraction = (event.clientX - rect.left) / rect.width;
    onIndexChange(Math.min(last, Math.max(0, Math.round(fraction * last))));
  };

  const current = byFrame.get(index);
  const anchors = [...new Set(promptFrames)].sort((a, b) => a - b);
  const sampleCount = Math.min(120, Math.max(1, nFrames));
  const bins = Array.from({ length: sampleCount }, () => ({ total: 0, score: 0, present: 0, prompt: false }));
  for (const record of scores) {
    const bin = bins[Math.min(sampleCount - 1, Math.floor(record.frame_index * sampleCount / Math.max(1, nFrames)))];
    if (bin) {
      bin.total++;
      if (record.object_present) {
        bin.present++;
        bin.score += record.score;
      }
    }
  }
  for (const frame of promptFrames) {
    const bin = bins[Math.min(sampleCount - 1, Math.floor(frame * sampleCount / Math.max(1, nFrames)))];
    if (bin) bin.prompt = true;
  }
  const thumbnails = Array.from({ length: Math.min(10, nFrames) }, (_, slot) =>
    Math.round(slot * last / Math.max(1, Math.min(10, nFrames) - 1)),
  );
  const previousAnchor = [...anchors].reverse().find((frame) => frame < index);
  const nextAnchor = anchors.find((frame) => frame > index);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      onIndexChange(index >= last ? 0 : index + 1);
    }, 1000 / Math.max(1, fps));
    return () => clearInterval(timer);
  }, [playing, index, last, fps, onIndexChange]);

  return (
      <div className="timeline-shell rounded-panel border px-3.5 py-3">
      <div className="timeline-filmstrip mb-2" aria-label="Clip frame overview">
        {thumbnails.map((frame) => (
          <button key={frame} type="button" onClick={() => onIndexChange(frame)} aria-current={frame === index ? "true" : "false"} aria-label={`Go to frame ${frame + 1}`} title={`Frame ${frame + 1}`}>
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
          className="h-12 w-full cursor-pointer rounded-lg border border-ink-800 bg-ink-950/50"
          role="img"
          aria-label="Per-frame mask confidence"
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
            const absent = tracked && sample.present === 0;
            const confidence = sample.present ? sample.score / sample.present : 0;
            const height = tracked ? Math.max(0.04, confidence) : 0.02;
            const colour = !tracked
              ? "var(--color-ink-750)"
              : absent
                ? "var(--color-negative)"
                : sample.prompt
                  ? "var(--color-accent-400)"
                  : "var(--color-positive)";
            return (
              <rect
                key={bin}
                x={bin + 0.12}
                y={1 - height}
                width={0.76}
                height={height}
                fill={colour}
                opacity={tracked ? 0.85 : 0.4}
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

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <Button
          variant="secondary"
          className="px-2 py-1.5 text-[10px]"
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? "Pause clip" : "Play clip"}
          aria-pressed={playing}
          disabled={nFrames <= 1}
        >
          {playing ? "Ⅱ Pause" : "▶ Play"}
        </Button>
        <Button
          variant="ghost"
          className="px-2 py-1.5 text-[10px]"
          onClick={() => previousAnchor !== undefined && onIndexChange(previousAnchor)}
          disabled={previousAnchor === undefined}
          aria-label="Previous prompt frame"
        >
          ← Prompt
        </Button>
        <Button
          variant="ghost"
          className="px-2 py-1.5 text-[10px]"
          onClick={() => nextAnchor !== undefined && onIndexChange(nextAnchor)}
          disabled={nextAnchor === undefined}
          aria-label="Next prompt frame"
        >
          Prompt →
        </Button>
        <span className="ml-auto font-mono text-[10px] text-ink-500">
          {anchors.length ? `${anchors.length} anchor${anchors.length === 1 ? "" : "s"}` : "No prompt anchors"}
        </span>
      </div>

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
              <span className={current.object_present ? "text-positive" : "text-negative"}>{current.object_present ? "Object present" : "Object absent"}</span>
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

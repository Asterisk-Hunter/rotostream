"use client";

import { useCallback, useRef, useState } from "react";

import { formatTimecode } from "@/lib/format";
import type { PointPrompt } from "@/lib/types";
import { frameCoordinates } from "@/lib/coordinates";

import { CursorIcon } from "./icons";
import { Badge, Button, Spinner, cx } from "./ui";

interface Props {
  frameUrl: string;
  overlayUrl: string | null;
  width: number;
  height: number;
  frameIndex: number;
  nFrames: number;
  fps: number;
  prompts: PointPrompt[];
  busy?: boolean;
  disabled?: boolean;
  hint?: string;
  onAddPoint: (point: PointPrompt) => void;
}

/**
 * Click coordinates are converted from rendered pixels back to source-frame pixels,
 * which is the only coordinate space the API and the contract speak.
 */
export function FrameStage({
  frameUrl,
  overlayUrl,
  width,
  height,
  frameIndex,
  nFrames,
  fps,
  prompts,
  busy,
  disabled,
  hint,
  onAddPoint,
}: Props) {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const [foreground, setForeground] = useState(true);
  const [keyboardPoint, setKeyboardPoint] = useState({ x: 0.5, y: 0.5 });
  const [focused, setFocused] = useState(false);

  const toFrameCoords = useCallback(
    (clientX: number, clientY: number) => {
      const element = surfaceRef.current;
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      return frameCoordinates(clientX, clientY, rect, width, height);
    },
    [width, height],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="checker relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-panel border border-ink-700 p-3">
        <div
          className="relative"
          style={{ aspectRatio: `${width} / ${height}`, maxWidth: "100%", maxHeight: "100%" }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frameUrl}
            alt={`Frame ${frameIndex}`}
            width={width}
            height={height}
            draggable={false}
            className="block h-full w-full select-none rounded-[3px]"
          />

          {overlayUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={overlayUrl}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full select-none rounded-[3px]"
            />
          )}

          {prompts.map((point, index) => (
            <span
              key={`${point.x}-${point.y}-${index}`}
              className={cx(
                "pointer-events-none absolute flex h-4 w-4 -translate-x-1/2 -translate-y-1/2 items-center justify-center",
                "rounded-full border font-mono text-[10px] leading-none shadow-[0_0_0_1px_rgba(0,0,0,0.55)]",
                point.positive
                  ? "border-positive bg-positive/25 text-positive"
                  : "border-negative bg-negative/25 text-negative",
              )}
              style={{ left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` }}
            >
              {point.positive ? "+" : "−"}
            </span>
          ))}

          <div
            ref={surfaceRef}
            role="button"
            tabIndex={disabled ? -1 : 0}
            aria-disabled={disabled}
            aria-label="Prompt object on frame. Arrow keys move the cursor; Enter marks the object; Shift Enter excludes background."
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={(event) => {
              if (disabled) return;
              const steps: Record<string, { x: number; y: number }> = {
                ArrowLeft: { x: -0.02, y: 0 }, ArrowRight: { x: 0.02, y: 0 },
                ArrowUp: { x: 0, y: -0.02 }, ArrowDown: { x: 0, y: 0.02 },
              };
              const step = steps[event.key];
              if (step) {
                event.preventDefault(); event.stopPropagation();
                setKeyboardPoint((point) => ({
                  x: Math.min(1, Math.max(0, point.x + step.x)),
                  y: Math.min(1, Math.max(0, point.y + step.y)),
                }));
              } else if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onAddPoint({
                  x: Math.min(width - 1, Math.round(keyboardPoint.x * width)),
                  y: Math.min(height - 1, Math.round(keyboardPoint.y * height)),
                  positive: foreground && !event.shiftKey && !event.altKey,
                });
              }
            }}
            onContextMenu={(event) => event.preventDefault()}
            onPointerDown={(event) => {
              if (disabled || (event.button !== 0 && event.button !== 2)) return;
              const coords = toFrameCoords(event.clientX, event.clientY);
              if (!coords) return;
              event.preventDefault();
              // Left click adds foreground; right click, alt or shift adds background.
              const positive =
                foreground && event.button === 0 && !event.altKey && !event.shiftKey && !event.metaKey;
              onAddPoint({ ...coords, positive });
            }}
            className={cx(
              "absolute inset-0",
              disabled ? "cursor-default" : "cursor-crosshair",
            )}
          />
          {focused && !disabled && <span aria-hidden="true" className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-300 bg-accent-500/20" style={{ left: `${keyboardPoint.x * 100}%`, top: `${keyboardPoint.y * 100}%` }} />}
        </div>

        {!overlayUrl && !busy && !disabled && (
          <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center">
            <span className="flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-900/90 px-3 py-1.5 text-[11px] text-ink-300 backdrop-blur">
              <CursorIcon className="h-3 w-3 text-accent-400" />
              {hint ?? "Click the object to prompt it"}
            </span>
          </div>
        )}

        {busy && (
          <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-900/90 px-2.5 py-1 text-[11px] text-ink-300 backdrop-blur">
            <Spinner />
            segmenting
          </div>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 px-0.5">
        <div className="flex items-center gap-2">
          <Badge tone="neutral">
            frame <span className="tnum">{frameIndex + 1}</span> /{" "}
            <span className="tnum">{nFrames}</span>
          </Badge>
          <span className="tnum font-mono text-[10px] text-ink-400">
            {formatTimecode(frameIndex, fps)}
          </span>
        </div>
        <span className="tnum font-mono text-[10px] text-ink-500">
          {width}×{height} working resolution
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button variant={foreground ? "primary" : "secondary"} disabled={disabled} aria-pressed={foreground} onClick={() => setForeground(true)}>Mark object</Button>
        <Button variant={!foreground ? "primary" : "secondary"} disabled={disabled} aria-pressed={!foreground} onClick={() => setForeground(false)}>Exclude background</Button>
        <span className="text-[10px] text-ink-400">← → scrub · focus frame + Enter to prompt</span>
      </div>
    </div>
  );
}

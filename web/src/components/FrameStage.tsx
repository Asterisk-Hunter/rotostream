"use client";

import { useCallback, useRef, useState } from "react";

import { frameBox, frameCoordinates } from "@/lib/coordinates";
import { formatTimecode } from "@/lib/format";
import type { BoxPrompt, PointPrompt } from "@/lib/types";

import { Button, cx } from "./ui";

/** Fixed overlay strength: strong enough to judge the outline, light enough to see pixels. */
const MASK_OPACITY = 0.72;

interface Props {
  frameUrl: string;
  overlayUrl: string | null;
  width: number;
  height: number;
  frameIndex: number;
  nFrames: number;
  fps: number;
  prompts: PointPrompt[];
  box: BoxPrompt | null;
  busy?: boolean;
  disabled?: boolean;
  /** What the overlay on screen is: the tracked mask, or your clicks so far. */
  hint?: string;
  onAddPoint: (point: PointPrompt) => void;
  onAddBox: (box: BoxPrompt) => void;
  onRemovePoint: (index: number) => void;
  onRemoveBox: () => void;
}

type Tool = "object" | "exclude" | "box";

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
  box,
  busy,
  disabled,
  hint,
  onAddPoint,
  onAddBox,
  onRemovePoint,
  onRemoveBox,
}: Props) {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const [tool, setTool] = useState<Tool>("object");
  const [dragBox, setDragBox] = useState<{ start: { x: number; y: number }; end: { x: number; y: number } } | null>(null);
  const [keyboardPoint, setKeyboardPoint] = useState({ x: 0.5, y: 0.5 });
  const [focused, setFocused] = useState(false);
  const [showMask, setShowMask] = useState(true);

  const toFrameCoords = useCallback(
    (clientX: number, clientY: number) => {
      const element = surfaceRef.current;
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      return frameCoordinates(clientX, clientY, rect, width, height);
    },
    [width, height],
  );

  const boxStyle = (selection: BoxPrompt) => ({
    left: `${(Math.min(selection.x0, selection.x1) / width) * 100}%`,
    top: `${(Math.min(selection.y0, selection.y1) / height) * 100}%`,
    width: `${(Math.abs(selection.x1 - selection.x0) / width) * 100}%`,
    height: `${(Math.abs(selection.y1 - selection.y0) / height) * 100}%`,
  });

  const tools: Array<{ key: Tool; label: string; title: string }> = [
    { key: "object", label: "Object", title: "Click the object to keep it (foreground)" },
    { key: "exclude", label: "Exclude", title: "Click a region to remove it from the mask (background)" },
    { key: "box", label: "Box", title: "Drag a tight box around the object" },
  ];

  const instruction =
    tool === "box"
      ? "Drag a tight box around the object you want to keep."
      : tool === "exclude"
        ? "Click what the mask wrongly includes."
        : hint ?? "Click the object you want to track.";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="viewer-surface relative order-2 flex min-h-0 flex-1 items-center justify-center overflow-hidden p-5">
        <div
          className="relative h-full max-h-full w-auto max-w-full overflow-hidden rounded-[4px] shadow-[0_18px_48px_rgba(0,0,0,0.48)]"
          style={{ aspectRatio: `${width} / ${height}` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frameUrl}
            alt={`Frame ${frameIndex}`}
            width={width}
            height={height}
            draggable={false}
            className="block h-full w-full select-none object-contain"
          />

          {overlayUrl && showMask && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={overlayUrl}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain"
              style={{ opacity: MASK_OPACITY }}
            />
          )}

          {prompts.map((point, index) => (
            <button
              type="button"
              key={`${point.x}-${point.y}-${index}`}
              disabled={disabled}
              aria-label={`Remove ${point.positive ? "object" : "background"} prompt ${index + 1}`}
              title="Remove this click"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => { event.stopPropagation(); onRemovePoint(index); }}
              className={cx(
                "absolute z-20 flex h-5 w-5 -translate-x-1/2 -translate-y-1/2 cursor-pointer items-center justify-center",
                "rounded-full border font-mono text-[10px] leading-none shadow-[0_0_0_1px_rgba(0,0,0,0.55)]",
                point.positive
                  ? "border-positive bg-positive/25 text-positive"
                  : "border-negative bg-negative/25 text-negative",
              )}
              style={{ left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` }}
            >
              {point.positive ? "+" : "−"}
            </button>
          ))}

          {box && (
            <>
              <span aria-hidden="true" className="pointer-events-none absolute z-20 border-2 border-accent-300 bg-accent-400/10 shadow-[0_0_0_1px_rgba(2,10,20,0.72)]" style={boxStyle(box)} />
              <button
                type="button"
                disabled={disabled}
                aria-label="Remove object selection box"
                title="Remove box"
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => { event.stopPropagation(); onRemoveBox(); }}
                className="absolute z-30 -translate-y-full rounded bg-ink-950 px-1.5 py-0.5 text-[10px] text-ink-100 shadow"
                style={{ left: `${(Math.min(box.x0, box.x1) / width) * 100}%`, top: `${(Math.min(box.y0, box.y1) / height) * 100}%` }}
              >
                ×
              </button>
            </>
          )}

          {dragBox && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute border-2 border-dashed border-accent-200 bg-accent-400/15"
              style={boxStyle({ x0: dragBox.start.x, y0: dragBox.start.y, x1: dragBox.end.x, y1: dragBox.end.y })}
            />
          )}

          <div
            ref={surfaceRef}
            role="button"
            tabIndex={disabled ? -1 : 0}
            aria-disabled={disabled}
            aria-label={
              tool === "box"
                ? "Draw a box around the object on this frame."
                : "Prompt the object on this frame. Arrow keys move the cursor, Enter marks the object, Shift Enter excludes background."
            }
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
                  positive: tool === "object" && !event.shiftKey,
                });
              }
            }}
            onContextMenu={(event) => event.preventDefault()}
            onPointerDown={(event) => {
              if (disabled || (event.button !== 0 && event.button !== 2)) return;
              const coords = toFrameCoords(event.clientX, event.clientY);
              if (!coords) return;
              event.preventDefault();
              if (tool === "box" && event.button === 0) {
                event.currentTarget.setPointerCapture(event.pointerId);
                setDragBox({ start: coords, end: coords });
                return;
              }
              // Left click follows the selected tool; right click always excludes.
              const positive = tool === "object" && event.button === 0 && !event.altKey && !event.shiftKey;
              onAddPoint({ ...coords, positive });
            }}
            onPointerMove={(event) => {
              if (!dragBox) return;
              const coords = toFrameCoords(event.clientX, event.clientY);
              if (coords) setDragBox((current) => current ? { ...current, end: coords } : current);
            }}
            onPointerUp={(event) => {
              if (!dragBox) return;
              const end = toFrameCoords(event.clientX, event.clientY);
              const start = dragBox.start;
              setDragBox(null);
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
              if (!end) return;
              if (Math.abs(end.x - start.x) < 4 || Math.abs(end.y - start.y) < 4) {
                onAddPoint({ ...end, positive: true });
                return;
              }
              const selection = frameBox(start, end);
              if (selection) onAddBox(selection);
            }}
            className={cx(
              "absolute inset-0 z-10",
              disabled ? "cursor-default" : tool === "box" ? "cursor-crosshair" : "cursor-cell",
            )}
          />
          {focused && !disabled && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-300 bg-accent-500/20"
              style={{ left: `${keyboardPoint.x * 100}%`, top: `${keyboardPoint.y * 100}%` }}
            />
          )}
        </div>

        {busy && (
          <span
            role="status"
            className="pointer-events-none absolute right-4 top-3 font-mono text-[11px] text-accent-300"
          >
            segmenting…
          </span>
        )}
      </div>

      <div className="order-1 mb-2 flex flex-wrap items-center gap-2 border-b border-ink-800 pb-2">
        <div role="group" aria-label="Prompt tool" className="flex items-center gap-1">
          {tools.map((entry) => (
            <Button
              key={entry.key}
              variant={tool === entry.key ? "primary" : "ghost"}
              disabled={disabled}
              aria-pressed={tool === entry.key}
              title={entry.title}
              onClick={() => setTool(entry.key)}
              className="px-2.5 py-1.5 text-xs"
            >
              {entry.label}
            </Button>
          ))}
        </div>
        <p className="min-w-0 flex-1 truncate px-1 text-[11px] text-ink-400">{instruction}</p>
        <Button
          variant="ghost"
          disabled={!overlayUrl}
          aria-pressed={showMask}
          title="Show or hide the mask overlay"
          onClick={() => setShowMask((visible) => !visible)}
          className="px-2.5 py-1.5 text-xs"
        >
          {showMask ? "Mask on" : "Mask off"}
        </Button>
      </div>

      <div className="order-3 mt-2 flex flex-wrap items-center justify-between gap-3 px-0.5">
        <span className="flex items-center gap-2">
          <span className="text-xs text-ink-300">Frame</span>
          <span className="tnum font-mono text-xs text-ink-100">
            {frameIndex + 1} <span className="text-ink-500">/ {nFrames}</span>
          </span>
          <span className="tnum font-mono text-xs text-ink-400">{formatTimecode(frameIndex, fps)}</span>
        </span>
        <span className="tnum font-mono text-[11px] text-ink-500">
          {width} × {height} · {fps.toFixed(2)} fps · ← → frame
        </span>
      </div>
    </div>
  );
}

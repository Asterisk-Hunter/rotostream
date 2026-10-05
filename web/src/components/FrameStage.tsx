"use client";

import { useCallback, useRef, useState } from "react";

import { formatTimecode } from "@/lib/format";
import type { BoxPrompt, PointPrompt } from "@/lib/types";
import { frameBox, frameCoordinates } from "@/lib/coordinates";

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
  box: BoxPrompt | null;
  busy?: boolean;
  disabled?: boolean;
  hint?: string;
  onAddPoint: (point: PointPrompt) => void;
  onAddBox: (box: BoxPrompt) => void;
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
  box,
  busy,
  disabled,
  hint,
  onAddPoint,
  onAddBox,
}: Props) {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const [foreground, setForeground] = useState(true);
  const [tool, setTool] = useState<"point" | "box">("point");
  const [dragBox, setDragBox] = useState<{ start: { x: number; y: number }; end: { x: number; y: number } } | null>(null);
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

  const boxStyle = (selection: BoxPrompt) => ({
    left: `${(Math.min(selection.x0, selection.x1) / width) * 100}%`,
    top: `${(Math.min(selection.y0, selection.y1) / height) * 100}%`,
    width: `${(Math.abs(selection.x1 - selection.x0) / width) * 100}%`,
    height: `${(Math.abs(selection.y1 - selection.y0) / height) * 100}%`,
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-panel border border-ink-800 bg-ink-900 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.025)]">
        <div
          className="checker relative h-full max-h-full w-auto max-w-full overflow-hidden rounded-lg shadow-[0_24px_60px_rgba(0,0,0,0.42)]"
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

          {overlayUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={overlayUrl}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain"
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

          {box && (
            <span
              aria-label="Object selection box"
              className="pointer-events-none absolute border-2 border-accent-300 bg-accent-400/10 shadow-[0_0_0_1px_rgba(2,10,20,0.72)]"
              style={boxStyle(box)}
            />
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
            aria-label={tool === "box" ? "Draw a box around the object on this frame." : "Prompt object on frame. Arrow keys move the cursor; Enter marks the object; Shift Enter excludes background."}
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
              if (tool === "box" && event.button === 0) {
                event.currentTarget.setPointerCapture(event.pointerId);
                setDragBox({ start: coords, end: coords });
                return;
              }
              // Left click adds foreground; right click, alt or shift adds background.
              const positive =
                foreground && event.button === 0 && !event.altKey && !event.shiftKey && !event.metaKey;
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
              "absolute inset-0",
              disabled ? "cursor-default" : "cursor-crosshair",
            )}
          />
          {focused && !disabled && <span aria-hidden="true" className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-300 bg-accent-500/20" style={{ left: `${keyboardPoint.x * 100}%`, top: `${keyboardPoint.y * 100}%` }} />}
        </div>

        {!overlayUrl && !busy && !disabled && (
          <div className="pointer-events-none absolute inset-x-0 bottom-5 flex justify-center">
            <span className="flex items-center gap-1.5 rounded-full border border-ink-700/80 bg-ink-950/88 px-3 py-1.5 text-[11px] text-ink-200 shadow-lg backdrop-blur">
              <CursorIcon className="h-3 w-3 text-accent-400" />
              {tool === "box" ? "Drag a tight box around the object" : hint ?? "Click the object to prompt it"}
            </span>
          </div>
        )}

        {busy && (
          <div className="pointer-events-none absolute right-4 top-4 flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-900/90 px-2.5 py-1 text-[11px] text-ink-300 backdrop-blur">
            <Spinner />
            segmenting
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 px-0.5">
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
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <Button variant={tool === "point" && foreground ? "primary" : "secondary"} disabled={disabled} aria-pressed={tool === "point" && foreground} onClick={() => { setTool("point"); setForeground(true); }}>Mark object</Button>
        <Button variant={tool === "box" ? "primary" : "secondary"} disabled={disabled} aria-pressed={tool === "box"} onClick={() => { setTool("box"); setForeground(true); }}>Draw box</Button>
        <Button variant={tool === "point" && !foreground ? "primary" : "secondary"} disabled={disabled} aria-pressed={tool === "point" && !foreground} onClick={() => { setTool("point"); setForeground(false); }}>Exclude background</Button>
        <span className="text-[11px] text-ink-500">Start with a box · add clicks to correct · ← → scrub</span>
      </div>
    </div>
  );
}

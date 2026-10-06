"use client";

import { assetUrl } from "@/lib/api";
import { formatBytes, formatDuration, formatPercent, formatScore } from "@/lib/format";
import { exportPlan } from "@/lib/exportPlan";
import type { ReviewSummary } from "@/lib/review";
import { readiness } from "@/lib/review";
import {
  EXPORT_LABELS,
  EXPORT_ORDER,
  type BackgroundMode,
  type ExportKind,
  type ExportRecord,
  type Job,
  type ModelInfo,
  type Prompt,
  type Session,
  type TrackResult,
  type Video,
} from "@/lib/types";

import { AlertIcon, DownloadIcon, LayersIcon, PlayIcon, StopIcon, TrashIcon } from "./icons";
import { Button, EmptyState, Field, Meter, NumberInput, Panel, Select, Spinner, Toggle, cx } from "./ui";

const TONE_TEXT = {
  ok: "text-ink-300",
  warn: "text-warn",
  error: "text-negative",
} as const;

// -------------------------------------------------------------------- prompts
export function PromptPanel({
  prompts,
  frameIndex,
  preview,
  previewBusy,
  error,
  onRemovePoint,
  onUndo,
  canUndo,
  onRemoveBox,
  onClearFrame,
  onClearAll,
  model,
  acceptsBackgroundOnlyPrompts,
  restored = false,
}: {
  prompts: Prompt[];
  frameIndex: number;
  preview: { area: number; ratio: number } | null;
  previewBusy: boolean;
  error: string | null;
  onRemovePoint: (frameIndex: number, pointIndex: number) => void;
  onUndo: () => void;
  canUndo: boolean;
  onRemoveBox: () => void;
  onClearFrame: () => void;
  onClearAll: () => void;
  model: string;
  acceptsBackgroundOnlyPrompts: boolean;
  /** True when these prompts came back from the saved run rather than this session. */
  restored?: boolean;
}) {
  const current = prompts.find((prompt) => prompt.frame_index === frameIndex);
  const points = current?.points ?? [];
  const box = current?.box ?? null;
  const hasForeground = Boolean(box) || points.some((point) => point.positive);
  const backgroundOnlyFrames = prompts
    .filter((prompt) => !prompt.box && !prompt.points.some((point) => point.positive))
    .map((prompt) => prompt.frame_index);

  return (
    <Panel title="Prompts">
      <div className="space-y-3">
        <p className="text-xs leading-relaxed text-ink-300">
          Mark the object on any frame: click it, or drag a tight box around it. Add a background
          click where the mask spills. Each frame keeps its own prompts.
        </p>

        {restored && (
          <p className="text-[11px] leading-snug text-ink-400">
            These are the prompts that produced the current masks, restored from the saved run.
          </p>
        )}

        <div className="flex items-center justify-between">
          <span className="text-xs text-ink-300">
            Frame <span className="tnum font-mono text-ink-100">{frameIndex + 1}</span>
          </span>
          {current && (
            <Button variant="ghost" onClick={onClearFrame} className="px-2 py-1 text-[11px]">
              Clear frame
            </Button>
          )}
        </div>

        {!current ? (
          <p className="text-[11px] leading-relaxed text-ink-400">No prompts on this frame.</p>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {box && (
              <li>
                <button
                  type="button"
                  onClick={onRemoveBox}
                  title="Remove box"
                  aria-label="Remove box prompt"
                  className="rounded-[4px] border border-accent-400/40 bg-accent-500/10 px-2 py-1 font-mono text-[10px] text-accent-200 hover:bg-accent-500/20"
                >
                  box {Math.round(box.x1 - box.x0)}×{Math.round(box.y1 - box.y0)} · remove
                </button>
              </li>
            )}
            {points.map((point, index) => (
              <li key={`${point.x}-${point.y}-${index}`}>
                <button
                  type="button"
                  onClick={() => onRemovePoint(frameIndex, index)}
                  title="Remove this click"
                  aria-label={`Remove ${point.positive ? "foreground" : "background"} click at ${point.x}, ${point.y}`}
                  className={cx(
                    "flex items-center gap-1.5 rounded-[4px] border px-2 py-1.5 font-mono text-[11px] tnum transition-colors",
                    point.positive
                      ? "border-positive/40 bg-positive/10 text-positive hover:bg-positive/20"
                      : "border-negative/40 bg-negative/10 text-negative hover:bg-negative/20",
                  )}
                >
                  <span aria-hidden="true">{point.positive ? "+" : "−"}</span>
                  <span>
                    {point.x},{point.y}
                  </span>
                  <span aria-hidden="true">×</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {canUndo && (
          <Button variant="secondary" onClick={onUndo} className="w-full">
            Undo last prompt change
          </Button>
        )}

        {current && !hasForeground && (
          <p role="status" className="rounded-[6px] border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
            {acceptsBackgroundOnlyPrompts
              ? `${model} reads this frame as "no object here", so its mask is empty on purpose.`
              : `${model} needs a foreground click or a box on every prompted frame. Add one here, or clear this frame's prompts.`}
          </p>
        )}

        {backgroundOnlyFrames.length > 0 && hasForeground && (
          <p className="text-[11px] leading-snug text-ink-400">
            Frames with background-only prompts: {backgroundOnlyFrames.map((frame) => frame + 1).join(", ")}.
          </p>
        )}

        {prompts.length > 0 && (
          <div className="flex items-center justify-between border-t border-ink-800 pt-3 text-[11px] text-ink-400">
            <span>
              {prompts.length} frame{prompts.length === 1 ? "" : "s"} prompted
            </span>
            <Button variant="ghost" onClick={onClearAll} className="px-2 py-1 text-[11px]">
              <TrashIcon className="h-3.5 w-3.5" />
              Clear all
            </Button>
          </div>
        )}

        {previewBusy && (
          <p role="status" className="flex items-center gap-2 text-[11px] text-ink-300">
            <Spinner /> Updating preview…
          </p>
        )}

        {error && (
          <p className="flex items-start gap-1.5 rounded-[6px] border border-negative/30 bg-negative/10 px-2.5 py-2 text-[11px] leading-snug text-negative">
            <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
            {error}
          </p>
        )}

        {preview && (
          <div className="space-y-2 border-t border-ink-800 pt-3">
            <p className="tnum font-mono text-[11px] text-ink-300">
              Mask coverage on this frame: {formatPercent(preview.ratio, 1)}
            </p>
            {preview.ratio < 0.002 ? (
              <p role="status" className="text-[11px] leading-snug text-warn">
                That is very small — click the subject again, or draw a tighter box around it.
              </p>
            ) : preview.ratio > 0.85 ? (
              <p role="status" className="text-[11px] leading-snug text-warn">
                That covers most of the frame — add background clicks outside the subject.
              </p>
            ) : (
              <p role="status" className="text-[11px] leading-snug text-ink-400">
                Check the outline: click where it misses, or exclude what it spills onto.
              </p>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------- tracking
export function TrackPanel({
  models,
  model,
  onModelChange,
  bidirectional,
  onBidirectionalChange,
  canTrack,
  requirement,
  job,
  jobError,
  onTrack,
  onCancel,
  session,
  summary,
  promptsDirty,
  onReview,
  submitting = false,
  awaitingJob = false,
}: {
  models: ModelInfo[];
  model: string;
  onModelChange: (name: string) => void;
  bidirectional: boolean;
  onBidirectionalChange: (next: boolean) => void;
  canTrack: boolean;
  requirement: string | null;
  job: Job<TrackResult> | null;
  jobError: string | null;
  onTrack: () => void;
  onCancel: () => void;
  session: Session | null;
  summary: ReviewSummary | null;
  promptsDirty: boolean;
  onReview: () => void;
  submitting?: boolean;
  awaitingJob?: boolean;
}) {
  const selected = models.find((entry) => entry.name === model);
  const running = submitting || awaitingJob || job?.status === "queued" || job?.status === "running";
  const verdict = readiness(summary);

  return (
    <Panel title="Track">
      <div className="space-y-3">
        <Field label="Tracker">
          <Select value={model} disabled={running} onChange={(event) => onModelChange(event.target.value)}>
            {models.map((entry) => (
              <option key={entry.name} value={entry.name} disabled={!entry.implemented}>
                {entry.name}
                {entry.implemented ? "" : " · unavailable"}
              </option>
            ))}
          </Select>
        </Field>

        {selected?.implemented && (
          <p className="text-[11px] leading-snug text-ink-400">
            {selected.uses_memory
              ? "Memory tracker: follows one object across a real clip, including occlusions."
              : "Colour baseline: fast, but it re-segments every frame and does not keep object identity."}
          </p>
        )}

        {selected && !selected.implemented && (
          <p className="flex items-start gap-1.5 rounded-[6px] border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
            <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
            <span>
              {selected.error || "This tracker is registered but unavailable in this runtime."}
            </span>
          </p>
        )}

        <Toggle
          checked={bidirectional}
          onChange={onBidirectionalChange}
          label="Track both directions"
          hint="Forward after each prompt, then backward before the first one"
        />

        <div className="flex gap-2">
          <Button variant="primary" onClick={onTrack} disabled={!canTrack || running} className="flex-1">
            {running ? <Spinner className="border-ink-950/40 border-t-ink-950" /> : <PlayIcon />}
            {running ? "Tracking…" : "Track object"}
          </Button>
          {running && job && (
            <Button variant="danger" onClick={onCancel} title="Cancel this job" aria-label="Cancel tracking">
              <StopIcon />
            </Button>
          )}
        </div>

        {requirement && !running && (
          <p role="status" className="text-[11px] leading-snug text-ink-400">{requirement}</p>
        )}

        {job && (
          <div className="space-y-1.5">
            <Meter
              value={job.progress}
              tone={job.status === "failed" ? "negative" : job.status === "succeeded" ? "positive" : "accent"}
            />
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="truncate text-ink-300">{job.message || job.status}</span>
              <span className="tnum shrink-0 font-mono text-ink-400">{Math.round(job.progress * 100)}%</span>
            </div>
            {job.error && (
              <p className="rounded-[6px] border border-negative/30 bg-negative/10 px-2.5 py-2 font-mono text-[10px] leading-snug text-negative">
                {job.error}
              </p>
            )}
          </div>
        )}

        {job?.status === "failed" && (
          <p className="text-[11px] leading-snug text-ink-300">
            Nothing was saved. Fix the prompts described above, then track again.
          </p>
        )}

        {jobError && !job?.error && <p className="text-[11px] text-negative">{jobError}</p>}

        {promptsDirty && session && (
          <p role="status" className="rounded-[6px] border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
            Prompts changed since this run. Track again to apply them — the saved masks still
            reflect the previous prompts.
          </p>
        )}

        {session && summary && (
          <div className="space-y-3 border-t border-ink-800 pt-3">
            <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
              <Stat label="Frames with a mask" value={`${summary.nMasked} / ${summary.nFrames}`} />
              <Stat label="Low confidence" value={String(summary.lowConfidenceFrames.length)} />
              <Stat label="Lowest score" value={formatScore(summary.minScore)} />
              <Stat label="Run time" value={formatDuration(session.elapsed_s)} />
            </dl>

            {verdict && (
              <div
                role={verdict.tone === "ok" ? "status" : "alert"}
                className={cx(
                  "space-y-1.5 rounded-[6px] border px-3 py-2.5",
                  verdict.tone === "error"
                    ? "border-negative/40 bg-negative/10"
                    : verdict.tone === "warn"
                      ? "border-warn/40 bg-warn/10"
                      : "border-ink-700 bg-ink-850/50",
                )}
              >
                <p className={cx("text-[12px] font-medium", TONE_TEXT[verdict.tone])}>{verdict.headline}</p>
                <p className="text-[11px] leading-snug text-ink-300">{verdict.detail}</p>
                {verdict.tone !== "ok" && (
                  <Button variant="secondary" onClick={onReview} className="px-2.5 py-1.5 text-[11px]">
                    Review problem frames
                  </Button>
                )}
              </div>
            )}

            <p className="text-[11px] leading-snug text-ink-400">
              Tracker {session.model}
              {session.prompts?.length ? ` · ${session.prompts.length} prompted frame${session.prompts.length === 1 ? "" : "s"}` : ""}
              {" · "}run {formatDuration(session.elapsed_s)}
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] text-ink-400">{label}</dt>
      <dd className="tnum truncate font-mono text-xs text-ink-100">{value}</dd>
    </div>
  );
}

// --------------------------------------------------------------------- memory
export function MemoryPanel({ session, usesMemory }: { session: Session | null; usesMemory: boolean }) {
  const memory = session?.memory ?? {};
  const bankSize = typeof memory.bank_size === "number" ? memory.bank_size : null;
  const frames = Array.isArray(memory.frames) ? (memory.frames as number[]) : [];
  const detail = typeof memory.detail === "string" ? memory.detail : null;

  return (
    <Panel title="Memory bank">
      {!session ? (
        <EmptyState
          icon={<LayersIcon className="h-5 w-5" />}
          title="Nothing tracked yet"
          body="After a run this shows how many frames the tracker conditions on."
        />
      ) : (
        <div className="space-y-3">
          <p className="font-mono text-[11px] text-ink-300">
            {usesMemory ? "memory attention" : "memory-free"}
            {bankSize !== null ? ` · bank ${bankSize}` : ""}
          </p>
          {frames.length > 0 ? (
            <p className="font-mono text-[11px] leading-relaxed text-ink-300">
              frames {frames.slice(0, 24).join(", ")}
              {frames.length > 24 ? ` … +${frames.length - 24}` : ""}
            </p>
          ) : (
            <p className="text-[11px] leading-relaxed text-ink-400">
              {detail ?? "This tracker does not report memory contents."}
            </p>
          )}
          {detail && frames.length > 0 && (
            <p className="text-[11px] leading-relaxed text-ink-400">{detail}</p>
          )}
        </div>
      )}
    </Panel>
  );
}

// --------------------------------------------------------------------- export
export function ExportPanel({
  videoId,
  video,
  session,
  summary,
  kind,
  onKindChange,
  background,
  onBackgroundChange,
  blurRadius,
  onBlurRadiusChange,
  canExport,
  job,
  onExport,
  exports,
  onRefresh,
  promptsDirty,
  onReview,
  submitting = false,
  awaitingJob = false,
  jobError,
}: {
  videoId: string | null;
  video: Video | null;
  session: Session | null;
  summary: ReviewSummary | null;
  kind: ExportKind;
  onKindChange: (kind: ExportKind) => void;
  background: BackgroundMode;
  onBackgroundChange: (mode: BackgroundMode) => void;
  blurRadius: number;
  onBlurRadiusChange: (radius: number) => void;
  canExport: boolean;
  job: Job<TrackResult> | null;
  onExport: () => void;
  exports: ExportRecord[];
  onRefresh: () => void;
  promptsDirty: boolean;
  onReview: () => void;
  submitting?: boolean;
  awaitingJob?: boolean;
  jobError: string | null;
}) {
  const running = submitting || awaitingJob || job?.status === "queued" || job?.status === "running";
  const plan = exportPlan(kind, video, session, { background, blurRadius });

  return (
    <Panel
      title="Export"
      action={
        <Button variant="ghost" disabled={!videoId} onClick={onRefresh} className="px-2 py-1 text-[11px]">
          Refresh
        </Button>
      }
    >
      <div className="space-y-3">
        <Field label="Deliverable" hint={EXPORT_LABELS[kind].keeps}>
          <Select value={kind} onChange={(event) => onKindChange(event.target.value as ExportKind)}>
            {EXPORT_ORDER.map((entry) => (
              <option key={entry} value={entry}>
                {EXPORT_LABELS[entry].label}
              </option>
            ))}
          </Select>
        </Field>

        {kind === "replace_bg" && (
          <div className="grid grid-cols-2 gap-2">
            <Field label="Background">
              <Select
                value={background}
                onChange={(event) => onBackgroundChange(event.target.value as BackgroundMode)}
              >
                <option value="blur">Blurred</option>
                <option value="black">Black</option>
                <option value="white">White</option>
                <option value="green">Green screen</option>
              </Select>
            </Field>
            <Field label="Blur radius">
              <NumberInput
                min={1}
                max={128}
                disabled={background !== "blur"}
                value={blurRadius}
                onChange={(event) => onBlurRadiusChange(Number(event.target.value) || 24)}
              />
            </Field>
          </div>
        )}

        <div className="space-y-1.5 rounded-[6px] border border-ink-800 bg-ink-850/40 px-3 py-2.5">
          <p className="text-[12px] font-medium text-ink-100">What you will get</p>
          <ul className="space-y-1 text-[11px] leading-snug text-ink-300">
            {plan.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>

        {plan.incomplete && summary && (
          <p role="alert" className="space-y-1.5 rounded-[6px] border border-negative/40 bg-negative/10 px-2.5 py-2 text-[11px] leading-snug text-negative">
            <span className="block">
              {summary.absentFrames.length} frame
              {summary.absentFrames.length === 1 ? " has" : "s have"} no mask and will render as
              background only.
            </span>
            <Button variant="secondary" onClick={onReview} className="px-2.5 py-1.5 text-[11px]">
              Review those frames
            </Button>
          </p>
        )}

        {promptsDirty && (
          <p role="status" className="rounded-[6px] border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
            Prompts have changed since the last run, so this file will use the older masks. Track
            again first if you want your corrections included.
          </p>
        )}

        <Button variant="primary" onClick={onExport} disabled={!canExport || running} className="w-full">
          {running ? <Spinner className="border-ink-950/40 border-t-ink-950" /> : <DownloadIcon />}
          {running ? "Rendering…" : `Render ${EXPORT_LABELS[kind].label.toLowerCase()}`}
        </Button>

        {job && (
          <div className="space-y-1.5">
            <Meter
              value={job.progress}
              tone={job.status === "failed" ? "negative" : job.status === "succeeded" ? "positive" : "accent"}
            />
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="truncate text-ink-300">{job.message || job.status}</span>
              <span className="tnum shrink-0 font-mono text-ink-400">{Math.round(job.progress * 100)}%</span>
            </div>
            {job.error && (
              <p className="rounded-[6px] border border-negative/30 bg-negative/10 px-2.5 py-2 font-mono text-[10px] leading-snug text-negative">
                {job.error}
              </p>
            )}
          </div>
        )}

        {jobError && !job?.error && <p role="status" className="text-[11px] text-negative">{jobError}</p>}

        {exports.length > 0 && (
          <ul className="space-y-1.5 border-t border-ink-800 pt-3">
            {exports.map((record) => (
              <li key={record.id} className="space-y-1 rounded-[6px] border border-ink-800 bg-ink-850/40 px-2.5 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-[11px] text-ink-100">
                      {record.manifest?.label ?? EXPORT_LABELS[record.kind]?.label ?? record.kind}
                    </p>
                    <p className="tnum font-mono text-[10px] text-ink-400">
                      {record.status === "succeeded"
                        ? formatBytes(record.size_bytes)
                        : record.status}
                      {record.error ? ` · ${record.error}` : ""}
                    </p>
                  </div>
                  {record.status === "succeeded" && videoId ? (
                    <a
                      href={assetUrl.download(videoId, record.id)}
                      download
                      aria-label={`Download ${record.manifest?.label ?? record.kind}`}
                      className="inline-flex shrink-0 items-center gap-1 rounded-[5px] border border-ink-600 bg-ink-800 px-2 py-1 text-[10px] text-ink-100 transition-colors hover:bg-ink-750"
                    >
                      <DownloadIcon className="h-3 w-3" />
                      {EXPORT_LABELS[record.kind as ExportKind]?.ext ?? "file"}
                    </a>
                  ) : (
                    <span className="shrink-0 text-[10px] text-ink-400">{record.status}</span>
                  )}
                </div>
                {record.manifest?.notes?.length ? (
                  <p className="text-[10px] leading-snug text-ink-400">
                    {record.manifest.notes.slice(0, 2).join(" ")}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

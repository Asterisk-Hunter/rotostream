"use client";

import { assetUrl } from "@/lib/api";
import { formatBytes, formatDuration, formatPercent, formatScore } from "@/lib/format";
import {
  EXPORT_LABELS,
  type BackgroundMode,
  type ExportKind,
  type ExportRecord,
  type Job,
  type ModelInfo,
  type Prompt,
  type Session,
  type TrackResult,
} from "@/lib/types";

import { AlertIcon, DownloadIcon, LayersIcon, PlayIcon, StopIcon, TrashIcon } from "./icons";
import { Badge, Button, EmptyState, Field, Meter, NumberInput, Panel, Select, Spinner, Toggle, cx } from "./ui";

// -------------------------------------------------------------------- prompts
export function PromptPanel({
  prompts,
  frameIndex,
  preview,
  error,
  onRemovePoint,
  onClearFrame,
  onClearAll,
}: {
  prompts: Prompt[];
  frameIndex: number;
  preview: { area: number; ratio: number } | null;
  error: string | null;
  onRemovePoint: (frameIndex: number, pointIndex: number) => void;
  onClearFrame: () => void;
  onClearAll: () => void;
}) {
  const current = prompts.find((prompt) => prompt.frame_index === frameIndex);
  const points = current?.points ?? [];
  const totalPoints = prompts.reduce((sum, prompt) => sum + prompt.points.length, 0);

  return (
    <Panel
      title="Prompts"
      subtitle={`${totalPoints} click${totalPoints === 1 ? "" : "s"} across ${prompts.length} frame${prompts.length === 1 ? "" : "s"}`}
      action={
        prompts.length > 0 ? (
          <Button variant="ghost" onClick={onClearAll} className="px-1.5 py-1">
            <TrashIcon className="h-3.5 w-3.5" />
            clear all
          </Button>
        ) : null
      }
    >
      <div className="space-y-3">
        <div className="rounded-md border border-ink-700 bg-ink-800/50 px-2.5 py-2">
          <p className="text-[11px] text-ink-300">
            <span className="text-positive">Left click</span> marks the object ·{" "}
            <span className="text-negative">right click / alt-click</span> marks background to
            exclude.
          </p>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-wider text-ink-400">
            On frame <span className="tnum font-mono">{frameIndex + 1}</span>
          </span>
          {points.length > 0 && (
            <Button variant="ghost" onClick={onClearFrame} className="px-1.5 py-0.5 text-[10px]">
              clear frame
            </Button>
          )}
        </div>

        {points.length === 0 ? (
          <p className="text-[11px] leading-relaxed text-ink-400">
            No clicks on this frame yet.
          </p>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {points.map((point, index) => (
              <li key={`${point.x}-${point.y}-${index}`}>
                <button
                  type="button"
                  onClick={() => onRemovePoint(frameIndex, index)}
                  title="Remove this click"
                  aria-label={`Remove ${point.positive ? "foreground" : "background"} click at ${point.x}, ${point.y}`}
                  className={cx(
                    "group flex items-center gap-1.5 rounded-full border px-2 py-1 font-mono text-[10px] tnum transition-colors",
                    point.positive
                      ? "border-positive/40 bg-positive/10 text-positive hover:bg-positive/20"
                      : "border-negative/40 bg-negative/10 text-negative hover:bg-negative/20",
                  )}
                >
                  <span>{point.positive ? "+" : "−"}</span>
                  <span>
                    {point.x},{point.y}
                  </span>
                  <span className="opacity-0 transition-opacity group-hover:opacity-100">×</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {error && (
          <p className="flex items-start gap-1.5 rounded-md border border-negative/30 bg-negative/10 px-2.5 py-2 text-[11px] leading-snug text-negative">
            <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
            {error}
          </p>
        )}

        {preview && (
          <dl className="grid grid-cols-2 gap-2 border-t border-ink-700/70 pt-3">
            <div>
              <dt className="text-[10px] uppercase tracking-wider text-ink-400">Mask area</dt>
              <dd className="tnum font-mono text-xs text-ink-100">
                {preview.area.toLocaleString("en-US")} px
              </dd>
            </div>
            <div>
              <dt className="text-[10px] uppercase tracking-wider text-ink-400">Coverage</dt>
              <dd className="tnum font-mono text-xs text-ink-100">
                {formatPercent(preview.ratio, 2)}
              </dd>
            </div>
          </dl>
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
  job,
  jobError,
  onTrack,
  onCancel,
  session,
  submitting = false,
  awaitingJob = false,
}: {
  models: ModelInfo[];
  model: string;
  onModelChange: (name: string) => void;
  bidirectional: boolean;
  onBidirectionalChange: (next: boolean) => void;
  canTrack: boolean;
  job: Job<TrackResult> | null;
  jobError: string | null;
  onTrack: () => void;
  onCancel: () => void;
  session: Session | null;
  submitting?: boolean;
  awaitingJob?: boolean;
}) {
  const selected = models.find((entry) => entry.name === model);
  const running = submitting || awaitingJob || job?.status === "queued" || job?.status === "running";

  return (
    <Panel title="Track" subtitle={session ? `last run ${formatDuration(session.elapsed_s)}` : undefined}>
      <div className="space-y-3">
        <Field
          label="Tracker"
          hint={
            selected
              ? selected.description
              : "Choose a tracker to segment and follow an object."
          }
        >
          <Select value={model} disabled={running} onChange={(event) => onModelChange(event.target.value)}>
            {models.map((entry) => (
              <option key={entry.name} value={entry.name} disabled={!entry.implemented}>
                {entry.name}
                {entry.uses_memory ? " · memory" : ""}
                {entry.implemented ? "" : " · not implemented"}
              </option>
            ))}
          </Select>
        </Field>

        {selected?.implemented && (
          <div className="flex flex-wrap items-center gap-1.5">
            {selected.uses_memory && <Badge tone="accent">memory</Badge>}
            {selected.trainable && <Badge>trainable</Badge>}
            {selected.checkpoint_hint && (
              <span className="font-mono text-[10px] text-ink-400">{selected.checkpoint_hint}</span>
            )}
          </div>
        )}

        {selected && !selected.implemented && (
          <p className="flex items-start gap-1.5 rounded-md border border-warn/30 bg-warn/10 px-2.5 py-2 text-[11px] leading-snug text-warn">
            <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
            <span>
              <strong className="font-semibold">{selected.name}</strong>{" "}
              {selected.error
                ? `is unavailable (${selected.error}).`
                : "is registered but not implemented yet."}{" "}
              The contract is in <code className="font-mono">api/app/models/base.py</code> and{" "}
              <code className="font-mono">docs/MODEL_CONTRACT.md</code>.
            </span>
          </p>
        )}

        <Toggle
          checked={bidirectional}
          onChange={onBidirectionalChange}
          label="Track both directions"
          hint="Backwards from the first prompt, forwards from the last"
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

        {job && (
          <div className="space-y-1.5">
            <Meter
              value={job.progress}
              tone={job.status === "failed" ? "negative" : job.status === "succeeded" ? "positive" : "accent"}
            />
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="truncate text-ink-300">{job.message || job.status}</span>
              <span className="tnum shrink-0 font-mono text-ink-400">
                {Math.round(job.progress * 100)}%
              </span>
            </div>
            {job.error && (
              <p className="rounded-md border border-negative/30 bg-negative/10 px-2.5 py-2 font-mono text-[10px] leading-snug text-negative">
                {job.error}
              </p>
            )}
          </div>
        )}

        {jobError && !job?.error && (
          <p className="text-[11px] text-negative">{jobError}</p>
        )}

        {session && (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-2 border-t border-ink-700/70 pt-3">
            <Stat label="Frames tracked" value={`${session.n_tracked} / ${session.n_frames}`} />
            <Stat label="Mean score" value={formatScore(session.mean_score)} />
            <Stat
              label="Absent frames"
              value={session.absent_frames.length ? String(session.absent_frames.length) : "none"}
            />
            <Stat label="Model" value={session.model} />
          </dl>
        )}
      </div>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wider text-ink-400">{label}</dt>
      <dd className="tnum truncate font-mono text-xs text-ink-100">{value}</dd>
    </div>
  );
}

// --------------------------------------------------------------------- memory
export function MemoryPanel({
  session,
  usesMemory,
}: {
  session: Session | null;
  usesMemory: boolean;
}) {
  const memory = session?.memory ?? {};
  const bankSize = typeof memory.bank_size === "number" ? memory.bank_size : null;
  const frames = Array.isArray(memory.frames) ? (memory.frames as number[]) : [];
  const detail = typeof memory.detail === "string" ? memory.detail : null;

  return (
    <Panel title="Memory bank" subtitle="What the tracker conditions each frame on">
      {!session ? (
        <EmptyState
          icon={<LayersIcon className="h-5 w-5" />}
          title="Nothing tracked yet"
          body="After a run, this shows how many frames the tracker is attending to."
        />
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge tone={usesMemory ? "accent" : "neutral"}>
              {usesMemory ? "memory attention" : "memory-free"}
            </Badge>
            {bankSize !== null && (
              <span className="tnum font-mono text-[11px] text-ink-300">bank {bankSize}</span>
            )}
          </div>

          {frames.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {frames.slice(0, 24).map((frame) => {
                const prompted = session.prompt_frames.includes(frame);
                return (
                  <span
                    key={frame}
                    className={cx(
                      "rounded border px-1.5 py-0.5 font-mono text-[10px] tnum",
                      prompted
                        ? "border-accent-500/40 bg-accent-500/10 text-accent-300"
                        : "border-ink-600 bg-ink-800 text-ink-300",
                    )}
                  >
                    {frame}
                  </span>
                );
              })}
              {frames.length > 24 && (
                <span className="px-1 py-0.5 font-mono text-[10px] text-ink-400">
                  +{frames.length - 24}
                </span>
              )}
            </div>
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
  submitting = false,
  awaitingJob = false,
  jobError,
}: {
  videoId: string | null;
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
  submitting?: boolean;
  awaitingJob?: boolean;
  jobError: string | null;
}) {
  const running = submitting || awaitingJob || job?.status === "queued" || job?.status === "running";
  const kinds = Object.keys(EXPORT_LABELS) as ExportKind[];

  return (
    <Panel
      title="Export"
      subtitle="Deliverables rendered from the current session"
      action={
        <Button variant="ghost" disabled={!videoId} onClick={onRefresh} className="px-1.5 py-1 text-[10px]">
          refresh
        </Button>
      }
    >
      <div className="space-y-3">
        <Field label="Format" hint={EXPORT_LABELS[kind].hint}>
          <Select value={kind} onChange={(event) => onKindChange(event.target.value as ExportKind)}>
            {kinds.map((entry) => (
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

        <Button variant="secondary" onClick={onExport} disabled={!canExport || running} className="w-full">
          {running ? <Spinner /> : <DownloadIcon />}
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
              <span className="tnum shrink-0 font-mono text-ink-400">
                {Math.round(job.progress * 100)}%
              </span>
            </div>
            {job.error && (
              <p className="rounded-md border border-negative/30 bg-negative/10 px-2.5 py-2 font-mono text-[10px] leading-snug text-negative">
                {job.error}
              </p>
            )}
          </div>
        )}

        {jobError && !job?.error && <p role="status" className="text-[11px] text-negative">{jobError}</p>}

        {exports.length > 0 && (
          <ul className="space-y-1.5 border-t border-ink-700/70 pt-3">
            {exports.map((record) => (
              <li
                key={record.id}
                className="flex items-center justify-between gap-2 rounded-md border border-ink-700 bg-ink-800/40 px-2 py-1.5"
              >
                <div className="min-w-0">
                  <p className="truncate text-[11px] text-ink-100">
                    {EXPORT_LABELS[record.kind as ExportKind]?.label ?? record.kind}
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
                    aria-label={`Download ${EXPORT_LABELS[record.kind]?.label ?? record.kind}`}
                    className="inline-flex shrink-0 items-center gap-1 rounded-md border border-ink-600 bg-ink-800 px-2 py-1 text-[10px] text-ink-100 transition-colors hover:bg-ink-750"
                  >
                    <DownloadIcon className="h-3 w-3" />
                    {EXPORT_LABELS[record.kind as ExportKind]?.ext ?? "file"}
                  </a>
                ) : (
                  <Badge tone={record.status === "failed" ? "negative" : "warn"}>{record.status}</Badge>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

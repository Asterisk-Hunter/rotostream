"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { FrameStage } from "@/components/FrameStage";
import { ExportPanel, MemoryPanel, PromptPanel, TrackPanel } from "@/components/Panels";
import { ProjectRail, type UploadState } from "@/components/ProjectRail";
import { Timeline } from "@/components/Timeline";
import { AlertIcon, FilmIcon } from "@/components/icons";
import { Badge, EmptyState, Spinner, StatusDot } from "@/components/ui";
import { useJob } from "@/hooks/useJob";
import { api, assetUrl } from "@/lib/api";
import {
  type BackgroundMode,
  type ExportKind,
  type ExportRecord,
  type Health,
  type ModelInfo,
  type PointPrompt,
  type PreviewMask,
  type Prompt,
  type Session,
  type Video,
} from "@/lib/types";

const EMPTY_UPLOAD: UploadState = { active: false, progress: 0, error: null, filename: "" };

/**
 * Everything the studio needs for one clip, loaded as a unit.
 *
 * `id` is what makes this safe: when the selection changes, the previous clip no
 * longer matches, so the UI derives `null` instead of briefly rendering stale
 * frames. That is why nothing here is reset imperatively inside an effect.
 */
interface Clip {
  id: string;
  video: Video;
  session: Session | null;
  exports: ExportRecord[];
}

interface Preview extends PreviewMask {
  /** Frame + prompt signature this overlay was rendered for. */
  key: string;
}

/**
 * Pure fetch of everything belonging to one clip.
 *
 * Deliberately module-level and free of state writes: effects await this and then
 * assign the result, which keeps the effects' synchronous bodies side-effect free.
 */
async function fetchClip(videoId: string): Promise<Clip> {
  const record = await api.getVideo(videoId);
  if (record.status !== "ready") {
    return { id: videoId, video: record, session: null, exports: [] };
  }
  const [sessionResult, exportResult] = await Promise.allSettled([
    api.latestSession(record.id),
    api.listExports(record.id),
  ]);
  return {
    id: videoId,
    video: record,
    session: sessionResult.status === "fulfilled" ? sessionResult.value : null,
    exports: exportResult.status === "fulfilled" ? exportResult.value : [],
  };
}

export function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);

  const [frameIndex, setFrameIndex] = useState(0);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [editing, setEditing] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [exportKind, setExportKind] = useState<ExportKind>("alpha_webm");
  const [background, setBackground] = useState<BackgroundMode>("blur");
  const [blurRadius, setBlurRadius] = useState(24);
  const [bidirectional, setBidirectional] = useState(true);

  const [upload, setUpload] = useState<UploadState>(EMPTY_UPLOAD);
  const [extractJobId, setExtractJobId] = useState<string | null>(null);
  const [trackJobId, setTrackJobId] = useState<string | null>(null);
  const [exportJobId, setExportJobId] = useState<string | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const previewUrlRef = useRef<string | null>(null);
  const handledExtractRef = useRef<string | null>(null);
  const extractJob = useJob(extractJobId);
  const trackJob = useJob(trackJobId);
  const exportJob = useJob(exportJobId);

  // ------------------------------------------------------------- object urls
  // Releasing only touches a ref, so it is safe to call from an effect body or a
  // cleanup. State is never written imperatively just to invalidate an overlay:
  // `Preview.key` does that by comparison.
  const releasePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }, []);

  useEffect(() => releasePreview, [releasePreview]);

  // -------------------------------------------------------------- data loads
  const refreshVideos = useCallback(async () => {
    try {
      setVideos(await api.listVideos());
    } catch (cause) {
      setBootError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [healthPayload, modelList, videoList] = await Promise.all([
          api.health(),
          api.listModels(),
          api.listVideos(),
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setModels(modelList);
        setVideos(videoList);
        const preferred =
          modelList.find((entry) => entry.is_default && entry.implemented) ??
          modelList.find((entry) => entry.implemented);
        setModel(preferred?.name ?? healthPayload.default_model);
        if (videoList.length > 0) setSelectedId(videoList[0].id);
      } catch (cause) {
        if (cancelled) return;
        setBootError(
          cause instanceof Error
            ? `${cause.message} — is the API running on ${api.baseUrl}?`
            : String(cause),
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    void (async () => {
      try {
        const next = await fetchClip(selectedId);
        if (!cancelled) setClip(next);
      } catch (cause) {
        if (!cancelled) setBootError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // ------------------------------------------- reload once extraction finishes
  useEffect(() => {
    const job = extractJob.job;
    if (!job || !extractJob.done) return;
    // Job identity changes on every progress tick; this ref makes the refresh
    // happen exactly once per finished extraction.
    if (handledExtractRef.current === job.id) return;
    handledExtractRef.current = job.id;

    let cancelled = false;
    void (async () => {
      try {
        const [list, next] = await Promise.all([
          api.listVideos(),
          selectedId ? fetchClip(selectedId) : Promise.resolve(null),
        ]);
        if (cancelled) return;
        setVideos(list);
        if (next) setClip(next);
      } catch (cause) {
        if (!cancelled) setBootError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [extractJob.done, extractJob.job, selectedId]);

  // ---------------------------------------------- adopt the session after a run
  const trackStatus = trackJob.job?.status;
  useEffect(() => {
    if (trackStatus !== "succeeded" || !selectedId) return;
    let cancelled = false;
    void (async () => {
      try {
        const [sessionResult, exportResult, list] = await Promise.allSettled([
          api.latestSession(selectedId),
          api.listExports(selectedId),
          api.listVideos(),
        ]);
        if (cancelled) return;
        if (list.status === "fulfilled") setVideos(list.value);
        setClip((current) =>
          current && current.id === selectedId
            ? {
                ...current,
                session: sessionResult.status === "fulfilled" ? sessionResult.value : null,
                exports: exportResult.status === "fulfilled" ? exportResult.value : [],
              }
            : current,
        );
      } catch {
        /* the tracking result stays visible even if this refresh fails */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [trackStatus, selectedId]);

  // -------------------------------------------------- refresh the exports list
  const exportStatus = exportJob.job?.status;
  useEffect(() => {
    if (exportStatus !== "succeeded" || !selectedId) return;
    let cancelled = false;
    void (async () => {
      try {
        const next = await api.listExports(selectedId);
        if (cancelled) return;
        setClip((current) =>
          current && current.id === selectedId ? { ...current, exports: next } : current,
        );
      } catch {
        /* ignore: the user can hit refresh */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [exportStatus, selectedId]);

  // ------------------------------------------------------------ derived state
  const video = clip && clip.id === selectedId ? clip.video : null;
  const session = clip && clip.id === selectedId ? clip.session : null;
  const exportList = clip && clip.id === selectedId ? clip.exports : [];
  const ready = video?.status === "ready";
  const frames = video?.n_frames ?? 0;

  const currentPoints = useMemo(
    () => prompts.find((entry) => entry.frame_index === frameIndex)?.points ?? [],
    [prompts, frameIndex],
  );

  const promptSignature = useMemo(
    () => currentPoints.map((point) => `${point.positive ? "+" : "-"}${point.x},${point.y}`).join("|"),
    [currentPoints],
  );

  // Identifies exactly which frame + prompt set an overlay belongs to.
  const previewKey =
    video && currentPoints.length > 0 ? `${video.id}#${frameIndex}#${promptSignature}` : null;

  const freshPreview = preview && previewKey !== null && preview.key === previewKey ? preview : null;

  // ------------------------------------------------------------- live preview
  useEffect(() => {
    if (!previewKey || !ready || !video) return;
    let cancelled = false;

    const timer = setTimeout(() => {
      void (async () => {
        setPreviewBusy(true);
        try {
          const next = await api.previewMask(
            video.id,
            { frame_index: frameIndex, points: currentPoints, box: null },
            model || undefined,
          );
          if (cancelled) {
            URL.revokeObjectURL(next.url);
            return;
          }
          releasePreview();
          previewUrlRef.current = next.url;
          setPreview({ ...next, key: previewKey });
          setPreviewError(null);
        } catch (cause) {
          if (!cancelled) setPreviewError(cause instanceof Error ? cause.message : String(cause));
        } finally {
          if (!cancelled) setPreviewBusy(false);
        }
      })();
    }, 160);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [previewKey, ready, video, frameIndex, currentPoints, model, releasePreview]);

  // ----------------------------------------------------------- keyboard nav
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (!ready || !video) return;
      if (event.key === "ArrowRight") {
        setFrameIndex((index) => Math.min(video.n_frames - 1, index + 1));
        event.preventDefault();
      } else if (event.key === "ArrowLeft") {
        setFrameIndex((index) => Math.max(0, index - 1));
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [ready, video]);

  // ------------------------------------------------------------------ actions
  /** Selecting a clip is an event, so the per-clip edit state resets here. */
  const beginClip = useCallback(
    (videoId: string) => {
      setSelectedId(videoId);
      setFrameIndex(0);
      setPrompts([]);
      setEditing(false);
      setPreviewError(null);
      setExtractJobId(null);
      setTrackJobId(null);
      releasePreview();
    },
    [releasePreview],
  );

  const handleUpload = useCallback(
    async (file: File) => {
      setUpload({ active: true, progress: 0, error: null, filename: file.name });
      try {
        const result = await api.uploadVideo(file, (fraction) =>
          setUpload((state) => ({ ...state, progress: fraction })),
        );
        setUpload(EMPTY_UPLOAD);
        await refreshVideos();
        beginClip(result.video.id);
        setExtractJobId(result.job_id);
      } catch (cause) {
        setUpload({
          active: false,
          progress: 0,
          filename: file.name,
          error: cause instanceof Error ? cause.message : String(cause),
        });
      }
    },
    [refreshVideos, beginClip],
  );

  const handleDelete = useCallback(
    async (videoId: string) => {
      try {
        await api.deleteVideo(videoId);
        setVideos((list) => list.filter((entry) => entry.id !== videoId));
        if (selectedId === videoId) setSelectedId(null);
      } catch (cause) {
        setBootError(cause instanceof Error ? cause.message : String(cause));
      }
    },
    [selectedId],
  );

  const addPoint = useCallback(
    (point: PointPrompt) => {
      setEditing(true);
      setPrompts((list) => {
        const existing = list.find((entry) => entry.frame_index === frameIndex);
        if (!existing) {
          return [...list, { frame_index: frameIndex, points: [point], box: null }].sort(
            (a, b) => a.frame_index - b.frame_index,
          );
        }
        return list.map((entry) =>
          entry.frame_index === frameIndex ? { ...entry, points: [...entry.points, point] } : entry,
        );
      });
    },
    [frameIndex],
  );

  const removePoint = useCallback((targetFrame: number, pointIndex: number) => {
    setEditing(true);
    setPrompts((list) =>
      list
        .map((entry) =>
          entry.frame_index === targetFrame
            ? { ...entry, points: entry.points.filter((_, index) => index !== pointIndex) }
            : entry,
        )
        .filter((entry) => entry.points.length > 0),
    );
  }, []);

  const clearFrame = useCallback(() => {
    setEditing(true);
    setPrompts((list) => list.filter((entry) => entry.frame_index !== frameIndex));
  }, [frameIndex]);

  const clearAll = useCallback(() => {
    setEditing(false);
    setPrompts([]);
  }, []);

  const handleTrack = useCallback(async () => {
    if (!video || prompts.length === 0) return;
    try {
      const job = await api.startTracking(video.id, {
        prompts,
        model: model || undefined,
        bidirectional,
      });
      // Drop back to session overlays while the run is in flight: the freshly
      // tracked masks replace them the moment the job succeeds.
      setEditing(false);
      setTrackJobId(job.id);
    } catch (cause) {
      setBootError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [video, prompts, model, bidirectional]);

  const handleCancel = useCallback(async () => {
    if (!trackJobId) return;
    try {
      await api.cancelJob(trackJobId);
    } catch {
      /* the job may already have finished */
    }
  }, [trackJobId]);

  const handleExport = useCallback(async () => {
    if (!video) return;
    try {
      const job = await api.startExport(video.id, {
        kind: exportKind,
        session_id: session?.id,
        background,
        blur_radius: blurRadius,
      });
      setExportJobId(job.id);
    } catch (cause) {
      setBootError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [video, exportKind, session, background, blurRadius]);

  const refreshExports = useCallback(async () => {
    if (!selectedId) return;
    try {
      const next = await api.listExports(selectedId);
      setClip((current) =>
        current && current.id === selectedId ? { ...current, exports: next } : current,
      );
    } catch {
      /* ignore */
    }
  }, [selectedId]);

  // -------------------------------------------------------------- presentation
  const overlayUrl = useMemo(() => {
    if (!video) return null;
    if (!editing && session) return assetUrl.overlay(video.id, frameIndex, session.id);
    return freshPreview?.url ?? null;
  }, [video, editing, session, frameIndex, freshPreview]);

  const selectedModel = models.find((entry) => entry.name === model);

  return (
    <div className="mx-auto flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-ink-800 px-4 py-2.5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-sm font-semibold tracking-tight text-ink-100">
            Roto<span className="text-accent-400">Stream</span>
          </h1>
          <p className="hidden text-[11px] text-ink-400 sm:block">
            click once, propagate a mask across the clip
          </p>
        </div>

        <div className="flex items-center gap-3">
          {health ? (
            <>
              <Badge tone="accent">{health.device}</Badge>
              <span className="flex items-center gap-1.5 text-[10px] text-ink-400">
                <StatusDot
                  ok={health.ffmpeg}
                  title={health.ffmpeg ? "ffmpeg found" : "ffmpeg missing from PATH"}
                />
                ffmpeg
              </span>
              <span className="flex items-center gap-1.5 text-[10px] text-ink-400">
                <StatusDot
                  ok={health.torch}
                  title={health.torch ? "torch importable" : "torch not installed"}
                />
                torch
              </span>
              <span className="tnum font-mono text-[10px] text-ink-500">v{health.version}</span>
            </>
          ) : (
            !bootError && <Spinner />
          )}
        </div>
      </header>

      {bootError && (
        <div className="flex items-start gap-2 border-b border-negative/30 bg-negative/10 px-4 py-2 text-[11px] text-negative">
          <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
          <span className="font-mono leading-snug">{bootError}</span>
        </div>
      )}

      <div className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[240px_minmax(0,1fr)_330px] lg:overflow-hidden">
        <div className="min-h-0 lg:overflow-y-auto">
          <ProjectRail
            videos={videos}
            selectedId={selectedId}
            upload={upload}
            onSelect={beginClip}
            onUpload={handleUpload}
            onDelete={handleDelete}
          />
        </div>

        <main className="flex min-h-0 flex-col gap-3">
          {ready && video ? (
            <>
              <FrameStage
                frameUrl={assetUrl.frame(video.id, frameIndex)}
                overlayUrl={overlayUrl}
                width={video.frame_width}
                height={video.frame_height}
                frameIndex={frameIndex}
                nFrames={frames}
                fps={video.fps}
                prompts={currentPoints}
                busy={previewBusy}
                hint={
                  editing
                    ? "Left click to add a point, right click to exclude"
                    : session
                      ? "Showing tracked masks — click to refine"
                      : "Click the object to prompt it"
                }
                onAddPoint={addPoint}
              />
              <Timeline
                nFrames={frames}
                index={frameIndex}
                scores={session?.scores ?? []}
                promptFrames={session?.prompt_frames ?? []}
                onIndexChange={setFrameIndex}
              />
            </>
          ) : (
            <div className="checker flex min-h-0 flex-1 items-center justify-center rounded-panel border border-ink-700">
              {!video ? (
                <EmptyState
                  icon={<FilmIcon className="h-6 w-6" />}
                  title="No clip selected"
                  body="Upload a clip on the left, or pick one you already loaded."
                />
              ) : video.status === "failed" ? (
                <EmptyState
                  icon={<AlertIcon className="h-6 w-6" />}
                  title="Frame extraction failed"
                  body={video.error ?? undefined}
                />
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Spinner className="h-4 w-4" />
                  <p className="text-xs text-ink-300">
                    {extractJob.job?.message ?? "Extracting frames…"}
                  </p>
                  <p className="text-[11px] text-ink-400">
                    decoding to the configured working resolution
                  </p>
                </div>
              )}
            </div>
          )}
        </main>

        <aside className="flex min-h-0 flex-col gap-3 lg:overflow-y-auto lg:pr-0.5">
          <PromptPanel
            prompts={prompts}
            frameIndex={frameIndex}
            preview={freshPreview}
            error={previewError}
            onRemovePoint={removePoint}
            onClearFrame={clearFrame}
            onClearAll={clearAll}
          />
          <TrackPanel
            models={models}
            model={model}
            onModelChange={setModel}
            bidirectional={bidirectional}
            onBidirectionalChange={setBidirectional}
            canTrack={Boolean(ready && prompts.length > 0)}
            job={trackJob.job}
            jobError={trackJob.error}
            onTrack={handleTrack}
            onCancel={handleCancel}
            session={session}
          />
          <MemoryPanel session={session} usesMemory={selectedModel?.uses_memory ?? false} />
          <ExportPanel
            videoId={video?.id ?? null}
            kind={exportKind}
            onKindChange={setExportKind}
            background={background}
            onBackgroundChange={setBackground}
            blurRadius={blurRadius}
            onBlurRadiusChange={setBlurRadius}
            canExport={Boolean(session) && ready}
            job={exportJob.job}
            onExport={handleExport}
            exports={exportList}
            onRefresh={refreshExports}
          />
        </aside>
      </div>
    </div>
  );
}

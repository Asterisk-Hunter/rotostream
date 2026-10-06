"use client";

import { useRef, useState } from "react";

import { assetUrl } from "@/lib/api";
import { formatBytes, formatDuration, formatFrames } from "@/lib/format";
import type { Video } from "@/lib/types";

import { FilmIcon, TrashIcon, UploadIcon } from "./icons";
import { Button, EmptyState, Meter, Spinner, cx } from "./ui";

export interface UploadState {
  active: boolean;
  progress: number;
  error: string | null;
  filename: string;
}

interface Props {
  videos: Video[];
  selectedId: string | null;
  upload: UploadState;
  onSelect: (videoId: string) => void;
  onUpload: (file: File) => void;
  onDelete: (videoId: string) => void;
}

export function ProjectRail({ videos, selectedId, upload, onSelect, onUpload, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const pick = () => { if (!upload.active) inputRef.current?.click(); };

  const accept = (files: FileList | null) => {
    const file = files?.[0];
    if (file && !upload.active) onUpload(file);
  };

  return (
    <aside className="flex h-full min-h-0 flex-col gap-3">
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi"
        className="hidden"
        onChange={(event) => {
          accept(event.target.files);
          event.target.value = "";
        }}
      />

      <div className="flex items-center justify-between px-1">
        <span className="text-[13px] font-semibold text-ink-100">Clips</span>
        <span className="font-mono text-[11px] text-ink-400">{videos.length}</span>
      </div>

      <button
        type="button"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files);
        }}
        onClick={pick}
        aria-label="Upload a video clip"
        disabled={upload.active}
        className={cx(
          "clip-dropzone w-full cursor-pointer rounded-[6px] border border-dashed text-center transition-colors",
          dragging
            ? "border-accent-400 bg-accent-500/10"
            : "border-ink-600 bg-ink-850/40 hover:border-ink-500 hover:bg-ink-850",
        )}
      >
        <UploadIcon className="mx-auto h-4 w-4 text-ink-400" />
        <p className="mt-2.5 text-xs font-medium text-ink-200">
          {upload.active ? "Uploading…" : "Drop a clip or click to browse"}
        </p>
        <p className="mt-1 text-[11px] text-ink-500">MP4, MOV, WebM, MKV</p>
      </button>

      {upload.active && (
        <div className="space-y-1.5" role="status" aria-live="polite">
          <div className="flex items-center justify-between gap-2 text-[11px] text-ink-300">
            <span className="truncate">{upload.filename}</span>
            <span className="tnum font-mono">{Math.round(upload.progress * 100)}%</span>
          </div>
          <Meter value={upload.progress} />
        </div>
      )}

      {upload.error && (
        <p role="alert" className="rounded-[5px] border border-negative/30 bg-negative/10 px-2.5 py-2 text-xs text-negative">
          {upload.error}
        </p>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-0.5">
        {videos.length === 0 && !upload.active && (
          <EmptyState
            icon={<FilmIcon className="h-5 w-5" />}
            title="No clips yet"
            body="Uploaded videos land here. Frames are extracted to a working resolution and masks are tracked against those frames."
          />
        )}

        {videos.map((video) => {
          const selected = video.id === selectedId;
          return (
            <div
              key={video.id}
              className={cx(
                "clip-card group relative rounded-panel border transition-colors",
                selected
                  ? "border-accent-500/50 bg-accent-500/[0.08] shadow-[inset_2px_0_0_var(--color-accent-400)]"
                  : "border-ink-800 bg-ink-900/50 hover:border-ink-700",
              )}
            >
              <button
                type="button"
                onClick={() => { setDeleteId(null); onSelect(video.id); }}
                aria-pressed={selected}
                className="flex w-full gap-3 p-2.5 text-left"
              >
                  <span className="h-11 w-16 shrink-0 overflow-hidden rounded-[4px] border border-ink-700 bg-ink-800">
                  {video.status === "ready" ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={assetUrl.thumbnail(video.id)}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <span className="flex h-full items-center justify-center">
                      {video.status === "extracting" ? (
                        <Spinner />
                      ) : (
                        <FilmIcon className="h-4 w-4 text-ink-500" />
                      )}
                    </span>
                  )}
                </span>

                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-medium text-ink-100">{video.filename}</span>
                  {video.status === "ready" ? (
                    <span className="mt-1 block tnum font-mono text-[10px] text-ink-400">
                      {formatFrames(video.n_frames)} · {video.fps.toFixed(2)} fps ·{" "}
                      {formatDuration(video.duration_s)} · {video.frame_width}×{video.frame_height} ·{" "}
                      {formatBytes(video.size_bytes)}
                    </span>
                  ) : (
                    <span className="mt-1 block text-[11px] text-ink-400">
                      {video.status === "failed" ? video.error ?? "processing failed" : `${video.status}…`}
                    </span>
                  )}
                </span>
              </button>

              <div className="absolute right-1.5 top-1.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                <Button
                  variant="ghost"
                  aria-label={`Delete ${video.filename}`}
                  title="Delete clip and all its masks"
                  onClick={() => setDeleteId(video.id)}
                  className="px-1.5 py-1"
                >
                  <TrashIcon className="h-3.5 w-3.5" />
                </Button>
              </div>
              {deleteId === video.id && (
                <div className="space-y-2 border-t border-negative/25 bg-negative/5 p-2.5">
                  <p className="text-[11px] text-ink-200">Delete this clip, all masks and exports?</p>
                  <div className="flex gap-2">
                    <Button variant="danger" onClick={() => { setDeleteId(null); onDelete(video.id); }}>Delete clip</Button>
                    <Button variant="ghost" onClick={() => setDeleteId(null)}>Keep clip</Button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}

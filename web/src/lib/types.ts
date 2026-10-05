/**
 * TypeScript mirrors of the API schemas in api/app/schemas.py.
 * Keep these in sync by hand — the API is the source of truth.
 */

export type VideoStatus = "uploaded" | "extracting" | "ready" | "failed";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export type ExportKind =
  | "alpha_webm"
  | "overlay_mp4"
  | "cutout_zip"
  | "mask_zip"
  | "mask_rle_json"
  | "replace_bg";

export type BackgroundMode = "blur" | "black" | "white" | "green";

export interface Health {
  status: string;
  version: string;
  device: string;
  workspace: string;
  ffmpeg: boolean;
  torch: boolean;
  default_model: string;
}

export interface ModelInfo {
  name: string;
  description: string;
  uses_memory: boolean;
  trainable: boolean;
  implemented: boolean;
  checkpoint_hint: string;
  error: string;
  is_default: boolean;
}

export interface Video {
  id: string;
  filename: string;
  status: VideoStatus;
  width: number;
  height: number;
  fps: number;
  duration_s: number;
  n_frames: number;
  frame_width: number;
  frame_height: number;
  size_bytes: number;
  created_at: string;
  error: string | null;
  has_audio: boolean;
}

export interface UploadResult {
  video: Video;
  job_id: string;
}

export interface Job<Result = Record<string, unknown>> {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  message: string;
  error: string | null;
  created_at: string;
  updated_at: string;
  video_id: string | null;
  result: Result | null;
}

export interface FrameScore {
  frame_index: number;
  score: number;
  object_present: boolean;
  prompted: boolean;
}

export interface Session {
  id: string;
  video_id: string;
  model: string;
  created_at: string;
  n_tracked: number;
  n_frames: number;
  prompt_frames: number[];
  mean_score: number;
  absent_frames: number[];
  elapsed_s: number;
  cancelled: boolean;
  status: JobStatus;
  error: string | null;
  scores: FrameScore[];
  memory: Record<string, unknown>;
}

export interface PointPrompt {
  x: number;
  y: number;
  positive: boolean;
}

export interface BoxPrompt {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Prompt {
  frame_index: number;
  points: PointPrompt[];
  box: BoxPrompt | null;
}

export interface ExportRecord {
  id: string;
  video_id: string;
  session_id: string | null;
  kind: ExportKind;
  status: JobStatus;
  filename: string;
  size_bytes: number;
  download_url: string;
  error: string | null;
}

export interface TrackResult {
  session_id: string;
  video_id: string;
  model: string;
  n_tracked: number;
  n_frames: number;
  mean_score: number;
  prompt_frames: number[];
  absent_frames: number[];
  elapsed_s: number;
  memory: Record<string, unknown>;
}

export interface PreviewMask {
  /** Blob URL for the rendered overlay PNG. Callers must revoke it. */
  url: string;
  area: number;
  ratio: number;
}

export const EXPORT_LABELS: Record<ExportKind, { label: string; hint: string; ext: string }> = {
  alpha_webm: {
    label: "Transparent cutout",
    hint: "Keeps the selected subject and removes the background. Some desktop players ignore WebM transparency; import it into an editor or choose Edited MP4 to preview it.",
    ext: "webm",
  },
  overlay_mp4: {
    label: "Mask preview MP4",
    hint: "Original clip with the tracked mask tinted on top. Use this to inspect edges; it does not remove the background.",
    ext: "mp4",
  },
  replace_bg: {
    label: "Edited MP4",
    hint: "Keeps the selected subject and visibly blurs or replaces the background. Plays in standard video players.",
    ext: "mp4",
  },
  cutout_zip: { label: "RGBA PNG sequence", hint: "One cutout per frame, zipped", ext: "zip" },
  mask_zip: { label: "Mask PNG sequence", hint: "Binary masks, 0 / 255", ext: "zip" },
  mask_rle_json: {
    label: "Mask RLE (JSON)",
    hint: "COCO-style run-length encoding, for training pipelines",
    ext: "json",
  },
};

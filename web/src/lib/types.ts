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
  /** True when a background-only prompt frame is a valid instruction for this tracker. */
  accepts_background_only_prompts: boolean;
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
  /** The prompts this run was built from; restored so a later run keeps them. */
  prompts?: Prompt[];
  quality?: SessionQuality;
  low_confidence_frames?: number[];
  background_only_frames?: number[];
  coverage?: number;
  sound?: boolean;
}

/** The API's review summary for a finished run. */
export interface SessionQuality {
  threshold: number;
  n_frames: number;
  n_masked: number;
  coverage: number;
  absent_frames: number[];
  low_confidence_frames: number[];
  background_only_frames: number[];
  problem_frames: number[];
  mean_score: number;
  min_score: number;
  sound: boolean;
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

export interface ExportManifest {
  kind: ExportKind;
  label: string;
  container: string;
  video_codec: string;
  keeps: string;
  width: number;
  height: number;
  source_width: number;
  source_height: number;
  fps: number;
  n_frames: number;
  duration_s: number;
  has_audio: boolean;
  audio_preserved: boolean;
  alpha: boolean;
  options: Record<string, unknown>;
  session_id: string | null;
  model: string | null;
  notes: string[];
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
  manifest?: ExportManifest;
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

/**
 * Deliverables, described by what they keep. The detailed promise for a specific
 * clip (resolution, frame count, audio) comes from the API manifest so the text
 * before rendering and the file that arrives afterwards cannot disagree.
 */
export const EXPORT_LABELS: Record<ExportKind, { label: string; keeps: string; ext: string }> = {
  replace_bg: {
    label: "Edited clip",
    keeps: "subject only, background replaced — plays anywhere",
    ext: "mp4",
  },
  alpha_webm: {
    label: "Transparent cutout",
    keeps: "subject only, background transparent — composite it in an editor",
    ext: "webm",
  },
  overlay_mp4: {
    label: "Mask check",
    keeps: "original footage with the mask tinted on top — inspect the edges",
    ext: "mp4",
  },
  cutout_zip: {
    label: "Cutout PNG sequence",
    keeps: "one RGBA PNG per frame, for compositing",
    ext: "zip",
  },
  mask_zip: {
    label: "Mask PNG sequence",
    keeps: "binary masks (0 or 255), one per frame",
    ext: "zip",
  },
  mask_rle_json: {
    label: "Mask RLE",
    keeps: "COCO-style run-length masks, for training pipelines",
    ext: "json",
  },
};

/** Order the picker presents: the visual deliverables first, then mask data. */
export const EXPORT_ORDER: ExportKind[] = [
  "replace_bg",
  "alpha_webm",
  "overlay_mp4",
  "cutout_zip",
  "mask_zip",
  "mask_rle_json",
];

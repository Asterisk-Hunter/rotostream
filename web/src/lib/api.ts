/**
 * Typed client for the RotoStream API.
 *
 * The backend base URL comes from NEXT_PUBLIC_API_BASE_URL (see .env.example).
 */
import type {
  ExportKind,
  ExportRecord,
  Health,
  Job,
  ModelInfo,
  PreviewMask,
  Prompt,
  Session,
  TrackResult,
  UploadResult,
  Video,
} from "./types";

const RAW_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? (
  process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8010"
);
export const BASE_URL = RAW_BASE.replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
    if (payload.detail) return JSON.stringify(payload.detail);
  } catch {
    /* not JSON */
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init, headers, signal: init.signal ?? AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** Upload via XHR so the dropzone can show real progress. */
export function uploadVideo(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/api/videos`);
    xhr.timeout = 15 * 60 * 1000;

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResult);
        } catch {
          reject(new ApiError("Malformed upload response", xhr.status));
        }
        return;
      }
      let detail = `Upload failed (${xhr.status})`;
      try {
        const payload = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (typeof payload.detail === "string") detail = payload.detail;
      } catch {
        /* not JSON */
      }
      reject(new ApiError(detail, xhr.status));
    };

    xhr.onerror = () => reject(new ApiError("Network error during upload", 0));
    xhr.ontimeout = () => reject(new ApiError("Upload timed out. Check the connection and try again.", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", 0));
    xhr.send(form);
  });
}

/**
 * Run the tracker on a single prompted frame. The API answers with the rendered
 * overlay PNG and puts mask stats in headers, so this is one request per click.
 * The returned URL is an object URL — the caller must revoke it.
 */
export async function previewMask(
  videoId: string,
  prompt: Prompt,
  model?: string,
  signal?: AbortSignal,
): Promise<PreviewMask> {
  const response = await fetch(`${BASE_URL}/api/videos/${videoId}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(60_000)]) : AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
  const blob = await response.blob();
  return {
    url: URL.createObjectURL(blob),
    area: Number(response.headers.get("X-Mask-Area") ?? 0),
    ratio: Number(response.headers.get("X-Mask-Ratio") ?? 0),
  };
}

export const assetUrl = {
  frame: (videoId: string, index: number) => `${BASE_URL}/api/videos/${videoId}/frames/${index}`,
  thumbnail: (videoId: string) => `${BASE_URL}/api/videos/${videoId}/thumbnail`,
  overlay: (videoId: string, index: number, sessionId?: string | null) =>
    `${BASE_URL}/api/videos/${videoId}/overlays/${index}` +
    (sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""),
  download: (videoId: string, exportId: string) =>
    `${BASE_URL}/api/videos/${videoId}/exports/${exportId}/download`,
};

export const api = {
  baseUrl: BASE_URL,

  health: () => request<Health>("/api/health"),
  listModels: () => request<ModelInfo[]>("/api/models"),

  listVideos: () => request<Video[]>("/api/videos"),
  getVideo: (videoId: string) => request<Video>(`/api/videos/${videoId}`),
  deleteVideo: (videoId: string) =>
    request<void>(`/api/videos/${videoId}`, { method: "DELETE" }),
  uploadVideo,

  getJob: (jobId: string) => request<Job<TrackResult>>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  eventsUrl: (jobId: string) => `${BASE_URL}/api/jobs/${jobId}/events`,

  previewMask,
  startTracking: (
    videoId: string,
    body: { prompts: Prompt[]; model?: string; bidirectional?: boolean; session_id?: string },
  ) =>
    request<Job<TrackResult>>(`/api/videos/${videoId}/track`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listSessions: (videoId: string) => request<Session[]>(`/api/videos/${videoId}/sessions`),
  latestSession: (videoId: string) =>
    request<Session>(`/api/videos/${videoId}/sessions/latest`),

  listExports: (videoId: string) => request<ExportRecord[]>(`/api/videos/${videoId}/exports`),
  startExport: (
    videoId: string,
    body: { kind: ExportKind; session_id?: string; background?: string; blur_radius?: number },
  ) =>
    request<Job<TrackResult>>(`/api/videos/${videoId}/exports`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

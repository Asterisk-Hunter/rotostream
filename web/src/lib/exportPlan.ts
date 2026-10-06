/**
 * What an export will contain, described before it is rendered.
 *
 * The wording mirrors `api/app/video.py::build_manifest`, which produces the same
 * facts for the finished artifact. Anything the studio cannot know locally (the
 * model name recorded in the session, the exact notes) is left to the API manifest
 * shown on the completed file.
 */
import type { BackgroundMode, ExportKind, Session, Video } from "./types";
import { EXPORT_LABELS } from "./types";
import { reviewSummary } from "./review";

export interface ExportPlan {
  label: string;
  keeps: string;
  ext: string;
  notes: string[];
  /** True when frames with no mask will visibly lose the subject. */
  incomplete: boolean;
}

export function exportPlan(
  kind: ExportKind,
  video: Video | null,
  session: Session | null,
  options: { background: BackgroundMode; blurRadius: number },
): ExportPlan {
  const meta = EXPORT_LABELS[kind];
  const nFrames = video?.n_frames ?? 0;
  const fps = video?.fps ?? 0;
  const width = video?.frame_width ?? 0;
  const height = video?.frame_height ?? 0;
  const duration = fps > 0 ? nFrames / fps : 0;
  const notes: string[] = [];

  notes.push(
    fps > 0
      ? `${nFrames.toLocaleString("en-US")} frames at ${width}×${height}, ${fps.toFixed(2)} fps (${duration.toFixed(2)}s).`
      : "Frame geometry is not available yet.",
  );
  if (video && (video.width !== width || video.height !== height) && width > 0) {
    notes.push(
      `Rendered at the working resolution (${width}×${height}); the upload was ${video.width}×${video.height}, so it is not an original-resolution master.`,
    );
  }

  const isMedia = kind === "alpha_webm" || kind === "overlay_mp4" || kind === "replace_bg";
  if (isMedia) {
    notes.push(
      video?.has_audio
        ? "Audio: copied from the source clip."
        : "Audio: the source clip has no audio track.",
    );
    if (kind === "alpha_webm") {
      notes.push("Some desktop players ignore WebM alpha; import it into an editor to see the cutout.");
    }
  } else if (kind === "cutout_zip") {
    notes.push("Alpha channel only: no audio.");
  } else {
    notes.push("Masks only: no video and no audio.");
  }

  if (kind === "replace_bg") {
    const solid: Record<Exclude<BackgroundMode, "blur">, string> = {
      black: "solid black",
      white: "solid white",
      green: "solid green (#00B140)",
    };
    const detail =
      options.background === "blur"
        ? `blurred (Gaussian radius ${Math.max(1, Math.round(options.blurRadius)) | 1})`
        : solid[options.background];
    notes.push(`Background: ${detail}; the subject keeps the original pixels.`);
  }

  const summary = reviewSummary(session);
  const incomplete = Boolean(summary && summary.absentFrames.length > 0);
  if (summary) {
    notes.push(
      `Mask source: session ${session?.id ?? "latest"} (${session?.model ?? "unknown tracker"}) — ` +
        `${summary.nMasked} of ${summary.nFrames} frames have a mask, ` +
        `${summary.absentFrames.length} without one and ${summary.lowConfidenceFrames.length} below the confidence threshold.`,
    );
    if (summary.absentFrames.length || summary.lowConfidenceFrames.length) {
      notes.push(
        `Frames with no mask are rendered with the ${
          kind === "cutout_zip" || kind === "alpha_webm" || kind === "replace_bg"
            ? "background only"
            : "mask empty"
        }; review them in the timeline before delivering this file.`,
      );
    }
  }

  return { label: meta.label, keeps: meta.keeps, ext: meta.ext, notes, incomplete };
}

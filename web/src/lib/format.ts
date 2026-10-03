/** Small formatting helpers. Every number the UI shows goes through here. */

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? Math.round(value) : value.toFixed(1)} ${units[exponent]}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const rest = rounded % 60;
  return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
}

export function formatFrames(count: number): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? "frame" : "frames"}`;
}

export function formatPercent(fraction: number, digits = 0): string {
  if (!Number.isFinite(fraction)) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatScore(score: number): string {
  if (!Number.isFinite(score)) return "—";
  return score.toFixed(3);
}

export function formatTimecode(frameIndex: number, fps: number): string {
  if (!Number.isFinite(fps) || fps <= 0) return `#${frameIndex}`;
  const hundredths = Math.round((frameIndex / fps) * 100);
  const minutes = Math.floor(hundredths / 6000);
  const seconds = (hundredths % 6000) / 100;
  return `${minutes}:${seconds.toFixed(2).padStart(5, "0")}`;
}

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

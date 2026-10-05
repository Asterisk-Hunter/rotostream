/** Prompt points always reference a valid pixel, including clicks on an edge. */
export function frameCoordinates(clientX: number, clientY: number, rect: {
  left: number; top: number; width: number; height: number;
}, width: number, height: number): { x: number; y: number } | null {
  if (rect.width <= 0 || rect.height <= 0 || width <= 0 || height <= 0) return null;
  return {
    x: Math.min(width - 1, Math.max(0, Math.round(((clientX - rect.left) / rect.width) * width))),
    y: Math.min(height - 1, Math.max(0, Math.round(((clientY - rect.top) / rect.height) * height))),
  };
}

/** Normalizes two valid frame points into the non-zero rectangle required by the API. */
export function frameBox(start: { x: number; y: number }, end: { x: number; y: number }): {
  x0: number; y0: number; x1: number; y1: number;
} | null {
  const x0 = Math.min(start.x, end.x);
  const y0 = Math.min(start.y, end.y);
  const x1 = Math.max(start.x, end.x);
  const y1 = Math.max(start.y, end.y);
  return x0 === x1 || y0 === y1 ? null : { x0, y0, x1, y1 };
}

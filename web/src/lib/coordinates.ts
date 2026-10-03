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

# Example video

[`moving-square.mp4`](moving-square.mp4) is a four-second, 640 × 360, 15 FPS clip
generated from the repository's deterministic synthetic sequence. It contains a
red square crossing a textured background and needs no external media license.

1. Start the studio and upload this clip.
2. On frame 0, click the center of the red square (approximately x=37, y=179).
3. Propagate forward, inspect the confidence timeline, then export a mask ZIP or
   transparent WebM.

`naive` demonstrates the whole workflow without a GPU. Select `sam2_memory` once
PyTorch, transformers and the pretrained encoder are available to use the neural
tracker. This clip demonstrates plumbing; it is not a real-world quality benchmark.

Rebuild the clip with `pnpm demo`. The generator reports the exact prompt coordinates.

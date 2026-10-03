"""Create a small, deterministic example clip that needs no media download."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "ml"))

from rotostream_ml.synthetic import linear  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "samples" / "moving-square.mp4")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequence = linear(n_frames=60, width=640, height=360, size=64, speed=8, fps=15)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{sequence.width}x{sequence.height}",
            "-r", str(sequence.fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(args.output),
        ],
        input=sequence.frames.tobytes(), check=True, timeout=60,
    )
    prompt = sequence.prompt_for()
    point = prompt.points[0]
    print(json.dumps({"video": str(args.output), "frame": prompt.frame_index, "click": [point.x, point.y]}))


if __name__ == "__main__":
    main()

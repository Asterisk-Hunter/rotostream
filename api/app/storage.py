"""Workspace layout on disk.

::

    <workspace>/
      videos/<video_id>/
        source.<ext>              original upload, untouched
        meta.json                 probe + extraction results (VideoOut shape)
        frames/000000.jpg         working-resolution frames (masks match these)
        sessions/<session_id>/
          meta.json               SessionOut shape
          masks/000000.png        binary masks, 0 / 255
          scores.json             per-frame score + presence
        exports/<export_id>.<ext>
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .settings import get_settings


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.videos_root = self.root / "videos"
        self.videos_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- videos
    def video_dir(self, video_id: str) -> Path:
        return self.videos_root / video_id

    def frames_dir(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "frames"

    def meta_path(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "meta.json"

    def create_video(self, filename: str) -> str:
        video_id = new_id()
        directory = self.video_dir(video_id)
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        (directory / "sessions").mkdir(parents=True, exist_ok=True)
        (directory / "exports").mkdir(parents=True, exist_ok=True)
        self.write_meta(
            video_id,
            {
                "id": video_id,
                "filename": filename,
                "status": "uploaded",
                "created_at": utcnow(),
                "error": None,
            },
        )
        return video_id

    def source_path(self, video_id: str) -> Path | None:
        directory = self.video_dir(video_id)
        for candidate in sorted(directory.glob("source.*")):
            return candidate
        return None

    def write_meta(self, video_id: str, meta: dict[str, Any]) -> None:
        path = self.meta_path(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(path)

    def read_meta(self, video_id: str) -> dict[str, Any]:
        path = self.meta_path(video_id)
        if not path.exists():
            raise FileNotFoundError(f"unknown video {video_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def update_meta(self, video_id: str, **changes: Any) -> dict[str, Any]:
        meta = self.read_meta(video_id)
        meta.update(changes)
        self.write_meta(video_id, meta)
        return meta

    def list_videos(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for directory in self.videos_root.iterdir():
            if not directory.is_dir():
                continue
            try:
                records.append(self.read_meta(directory.name))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records

    def delete_video(self, video_id: str) -> bool:
        directory = self.video_dir(video_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    # ----------------------------------------------------------- sessions
    def session_dir(self, video_id: str, session_id: str) -> Path:
        return self.video_dir(video_id) / "sessions" / session_id

    def create_session(self, video_id: str) -> str:
        session_id = new_id()
        (self.session_dir(video_id, session_id) / "masks").mkdir(parents=True, exist_ok=True)
        return session_id

    def masks_dir(self, video_id: str, session_id: str) -> Path:
        return self.session_dir(video_id, session_id) / "masks"

    def mask_path(self, video_id: str, session_id: str, frame_index: int) -> Path:
        return self.masks_dir(video_id, session_id) / f"{frame_index:06d}.png"

    def write_session_meta(self, video_id: str, session_id: str, meta: dict[str, Any]) -> None:
        path = self.session_dir(video_id, session_id) / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_session_meta(self, video_id: str, session_id: str) -> dict[str, Any]:
        path = self.session_dir(video_id, session_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown session {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self, video_id: str) -> list[dict[str, Any]]:
        sessions_root = self.video_dir(video_id) / "sessions"
        out: list[dict[str, Any]] = []
        if not sessions_root.is_dir():
            return out
        for directory in sessions_root.iterdir():
            meta_path = directory / "meta.json"
            if meta_path.exists():
                try:
                    out.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return out

    def latest_session_id(self, video_id: str) -> str | None:
        sessions = self.list_sessions(video_id)
        return sessions[0]["id"] if sessions else None

    # ------------------------------------------------------------ exports
    def exports_dir(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "exports"

    def export_files_dir(self, video_id: str) -> Path:
        """Artifacts live in ``exports/files/`` so ``*.meta.json`` records stay separate
        from payloads that are themselves JSON (the RLE export)."""
        directory = self.exports_dir(video_id) / "files"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def export_file_path(self, video_id: str, export_id: str, suffix: str) -> Path:
        return self.export_files_dir(video_id) / f"{export_id}{suffix}"

    def export_meta_path(self, video_id: str, export_id: str) -> Path:
        return self.exports_dir(video_id) / f"{export_id}.meta.json"

    def write_export_meta(self, video_id: str, export_id: str, meta: dict[str, Any]) -> None:
        path = self.export_meta_path(video_id, export_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_export_meta(self, video_id: str, export_id: str) -> dict[str, Any]:
        path = self.export_meta_path(video_id, export_id)
        if not path.exists():
            raise FileNotFoundError(f"unknown export {export_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def update_export_meta(self, video_id: str, export_id: str, **changes: Any) -> dict[str, Any]:
        meta = self.read_export_meta(video_id, export_id)
        meta.update(changes)
        self.write_export_meta(video_id, export_id, meta)
        return meta

    def list_exports(self, video_id: str) -> list[dict[str, Any]]:
        directory = self.exports_dir(video_id)
        records: list[dict[str, Any]] = []
        if not directory.is_dir():
            return records
        for path in directory.glob("*.meta.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records

    def export_artifact(self, video_id: str, export_id: str) -> Path | None:
        meta = self.read_export_meta(video_id, export_id)
        filename = meta.get("filename")
        if not filename:
            return None
        path = self.export_files_dir(video_id) / filename
        return path if path.exists() else None


@lru_cache
def get_workspace() -> Workspace:
    return Workspace(get_settings().resolved_workspace)


def reset_workspace_cache() -> None:
    get_workspace.cache_clear()

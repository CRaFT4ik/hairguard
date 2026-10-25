"""hairlab.clipstore — where clips and their labels live.

Two separate concerns, kept apart on purpose:

  ClipStore   frames and metadata on disk, one directory per clip
  LabelStore  the human verdict for a clip, in SQLite

Labels are stored outside the clip directory and keyed by clip id, so
relabelling costs one UPDATE and never rewrites image data.
"""
from __future__ import annotations

import json
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

DEFAULT_ROOT = Path.home() / ".hairguard" / "clips"
DEFAULT_LABEL_DB = Path.home() / ".hairguard" / "labels.db"

_JPEG_QUALITY = 88
_META_NAME = "meta.json"
_FRAME_GLOB = "frame_*.jpg"


class ClipStore:
    """Clips on disk: ``<root>/<clip_id>/frame_000.jpg`` plus ``meta.json``."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def save(self, frames: Sequence[np.ndarray], meta: dict[str, Any]) -> str:
        """Write a clip and return its id. Satisfies the ClipSink protocol."""
        clip_id, _ = self.reserve()
        self.write(clip_id, frames, meta)
        return clip_id

    def reserve(self) -> tuple[str, Path]:
        """Claim an id and its directory. Cheap: one mkdir, no image data.

        Split out from :meth:`write` so a caller can take the id immediately
        and push the actual encoding onto another thread.
        """
        return self._reserve_clip_dir()

    def write(self, clip_id: str, frames: Sequence[np.ndarray], meta: dict[str, Any]) -> None:
        """Encode and write a reserved clip. Slow: this is the JPEG work."""
        clip_dir = self._clip_dir(clip_id)

        for index, frame in enumerate(frames):
            cv2.imwrite(
                str(clip_dir / f"frame_{index:03d}.jpg"),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
            )

        full_meta = dict(meta)
        full_meta["clip_id"] = clip_id
        full_meta["frame_count"] = len(frames)
        full_meta["created_at"] = datetime.now(timezone.utc).isoformat()
        # The metadata file is what marks a clip complete, so it is written
        # last: a clip interrupted half-way is skipped rather than served
        # with missing frames.
        (clip_dir / _META_NAME).write_text(
            json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def delete(self, clip_id: str) -> None:
        shutil.rmtree(self._clip_dir(clip_id), ignore_errors=True)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def clip_ids(self) -> list[str]:
        """Every complete clip, oldest first (ids are time-ordered)."""
        return sorted(
            p.name for p in self._root.iterdir()
            if p.is_dir() and (p / _META_NAME).exists()
        )

    def exists(self, clip_id: str) -> bool:
        return (self._clip_dir(clip_id) / _META_NAME).exists()

    def frame_count(self, clip_id: str) -> int:
        return len(list(self._clip_dir(clip_id).glob(_FRAME_GLOB)))

    def frame_path(self, clip_id: str, index: int) -> Path:
        return self._clip_dir(clip_id) / f"frame_{index:03d}.jpg"

    def read_frame(self, clip_id: str, index: int) -> np.ndarray:
        frame = cv2.imread(str(self.frame_path(clip_id, index)))
        if frame is None:
            raise FileNotFoundError(f"clip {clip_id} has no frame {index}")
        return frame

    def read_meta(self, clip_id: str) -> dict[str, Any]:
        return json.loads((self._clip_dir(clip_id) / _META_NAME).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clip_dir(self, clip_id: str) -> Path:
        # Guard against a crafted id escaping the store root.
        safe = Path(clip_id).name
        return self._root / safe

    def _reserve_clip_dir(self) -> tuple[str, Path]:
        """Claim a fresh clip directory, creating it so the id cannot be reused.

        The timestamp alone is not unique: the Windows clock does not advance
        on every call, so two clips saved back to back can render identically.
        A counter suffix is appended until the directory is genuinely new.
        """
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        for attempt in range(1000):
            clip_id = base if attempt == 0 else f"{base}-{attempt}"
            clip_dir = self._root / clip_id
            try:
                clip_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return clip_id, clip_dir
        raise RuntimeError("could not allocate a clip directory")


class LabelStore:
    """The human verdict per clip, in SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_LABEL_DB) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    clip_id     TEXT PRIMARY KEY,
                    label       TEXT NOT NULL,
                    labelled_at TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------

    def set(self, clip_id: str, label: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO labels (clip_id, label, labelled_at) VALUES (?, ?, ?)
                ON CONFLICT(clip_id) DO UPDATE SET label=excluded.label,
                                                   labelled_at=excluded.labelled_at
                """,
                (clip_id, label, datetime.now(timezone.utc).isoformat()),
            )

    def unset(self, clip_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM labels WHERE clip_id = ?", (clip_id,))

    def get(self, clip_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT label FROM labels WHERE clip_id = ?", (clip_id,)
            ).fetchone()
        return row[0] if row else None

    def labelled_ids(self) -> set[str]:
        with self._connect() as conn:
            return {r[0] for r in conn.execute("SELECT clip_id FROM labels")}

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                label: count
                for label, count in conn.execute(
                    "SELECT label, COUNT(*) FROM labels GROUP BY label"
                )
            }

    def last_labelled(self) -> Optional[str]:
        """Most recently labelled clip id — the target of an undo."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT clip_id FROM labels ORDER BY labelled_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

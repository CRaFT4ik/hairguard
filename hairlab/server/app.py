"""hairlab.server.app — HTTP API and static hosting for the labelling UI.

The app is built by a factory so tests can point it at a temporary clip
store instead of the real one.

Labelling is a queue: the UI asks for the next few unlabelled clips, shows
one and preloads the rest, then posts a verdict. Nothing here decides what a
clip contains — that is entirely the person's call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hairlab.capture import LABELS, LABEL_IDS
from hairlab.clipstore import ClipStore, LabelStore

_STATIC_DIR = Path(__file__).parent / "static"


class LabelRequest(BaseModel):
    clip_id: str = Field(min_length=1)
    label: str


def create_app(
    store: Optional[ClipStore] = None,
    labels: Optional[LabelStore] = None,
) -> FastAPI:
    clips = store if store is not None else ClipStore()
    verdicts = labels if labels is not None else LabelStore()

    app = FastAPI(title="HairLab", docs_url=None, redoc_url=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_clip(clip_id: str) -> str:
        if not clips.exists(clip_id):
            raise HTTPException(status_code=404, detail="Клип не найден")
        return clip_id

    def _describe(clip_id: str) -> dict:
        meta = clips.read_meta(clip_id)
        count = int(meta.get("frame_count") or clips.frame_count(clip_id))
        # The rate the loop actually achieved, not the one it was asked for —
        # playing back at the configured fps shows the clip at the wrong speed.
        fps = meta.get("fps_measured") or meta.get("fps_configured") or meta.get("fps") or 10.0
        return {
            "clip_id": clip_id,
            "frame_count": count,
            "fps": float(fps),
            "created_at": meta.get("created_at"),
            "frames": [f"/api/clips/{clip_id}/frames/{i}" for i in range(count)],
        }

    def _unlabelled() -> list[str]:
        done = verdicts.labelled_ids()
        return [cid for cid in clips.clip_ids() if cid not in done]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.get("/api/labels/definitions")
    def label_definitions() -> dict:
        """The taxonomy, so the UI and the dataset can never disagree."""
        return {"labels": LABELS}

    @app.get("/api/queue")
    def queue(limit: int = 3) -> dict:
        """The next unlabelled clips. The UI shows the first and preloads the rest."""
        limit = max(1, min(limit, 10))
        pending = _unlabelled()
        return {
            "clips": [_describe(cid) for cid in pending[:limit]],
            "remaining": len(pending),
        }

    @app.get("/api/stats")
    def stats() -> dict:
        total = len(clips.clip_ids())
        counts = verdicts.counts()
        labelled = sum(counts.values())
        return {
            "total": total,
            "labelled": labelled,
            "remaining": max(0, total - labelled),
            "counts": counts,
            "can_undo": verdicts.last_labelled() is not None,
        }

    @app.post("/api/labels")
    def set_label(request: LabelRequest) -> dict:
        if request.label not in LABEL_IDS:
            raise HTTPException(status_code=422, detail=f"Неизвестная метка: {request.label}")
        _require_clip(request.clip_id)
        verdicts.set(request.clip_id, request.label)
        return {"ok": True, "clip_id": request.clip_id, "label": request.label}

    @app.post("/api/labels/undo")
    def undo() -> dict:
        clip_id = verdicts.last_labelled()
        if clip_id is None:
            raise HTTPException(status_code=404, detail="Отменять нечего")
        verdicts.unset(clip_id)
        return {"ok": True, "clip_id": clip_id}

    @app.get("/api/clips/{clip_id}/frames/{index}")
    def frame(clip_id: str, index: int) -> FileResponse:
        _require_clip(clip_id)
        path = clips.frame_path(clip_id, index)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Кадр не найден")
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    return app

"""FastAPI application and local dashboard routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from industrial_safety.events.types import EventStatus, SafetyEvent
from industrial_safety.storage import SQLiteEventStore

CameraHealthProvider = Callable[[], dict[str, dict[str, Any]]]


class ReviewRequest(BaseModel):
    reason: str | None = None


def create_app(
    store: SQLiteEventStore,
    evidence_root: str | Path,
    *,
    camera_health_provider: CameraHealthProvider | None = None,
    web_root: str | Path | None = None,
) -> FastAPI:
    evidence = Path(evidence_root).resolve()
    root = (
        Path(web_root).resolve()
        if web_root
        else Path(__file__).resolve().parents[3] / "web"
    )
    app = FastAPI(title="Industrial AI Safety Compliance Platform")
    app.mount("/static", StaticFiles(directory=root / "static"), name="static")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(root / "templates" / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        cameras = camera_health_provider() if camera_health_provider else {}
        return {
            "status": "ok",
            "service": "industrial-safety",
            "time": datetime.now().astimezone().isoformat(),
            "cameras": len(cameras),
        }

    @app.get("/api/cameras")
    async def cameras() -> dict[str, dict[str, Any]]:
        return camera_health_provider() if camera_health_provider else {}

    @app.get("/api/events")
    async def events(
        limit: int = Query(default=100, ge=1, le=1000),
        status: EventStatus | None = None,
    ) -> list[dict[str, Any]]:
        return [_serialize(event) for event in store.list_events(limit=limit, status=status)]

    @app.get("/api/events/{event_id}")
    async def event(event_id: str) -> dict[str, Any]:
        stored = store.get(event_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="event not found")
        return _serialize(stored)

    @app.post("/api/events/{event_id}/resolve")
    async def resolve(event_id: str, request: ReviewRequest) -> dict[str, Any]:
        return _review(store, event_id, EventStatus.RESOLVED, request.reason)

    @app.post("/api/events/{event_id}/dismiss")
    async def dismiss(event_id: str, request: ReviewRequest) -> dict[str, Any]:
        return _review(store, event_id, EventStatus.DISMISSED, request.reason)

    @app.get("/api/events/{event_id}/evidence", name="event_evidence")
    async def event_evidence(event_id: str) -> FileResponse:
        stored = store.get(event_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="event not found")
        candidate = _evidence_file(evidence, stored.snapshot_path)
        return Response(candidate.read_bytes(), media_type="image/jpeg")

    @app.get("/api/events/{event_id}/clip", name="event_clip")
    async def event_clip(event_id: str, request: Request) -> Response:
        stored = store.get(event_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="event not found")
        if stored.clip_path is None:
            raise HTTPException(status_code=404, detail="evidence clip not available")
        candidate = _evidence_file(evidence, stored.clip_path)
        return _video_response(candidate, request.headers.get("range"))

    @app.get("/api/stats")
    async def stats() -> dict[str, int]:
        values = store.stats()
        values["zone_entries"] = values.get("LOADING_AREA_ENTRY", 0) + values.get(
            "RESTRICTED_AREA_ENTRY", 0
        )
        return values

    return app


def _review(
    store: SQLiteEventStore,
    event_id: str,
    status: EventStatus,
    reason: str | None,
) -> dict[str, Any]:
    stored = store.update_status(event_id, status, reason=reason)
    if stored is None:
        raise HTTPException(status_code=404, detail="event not found")
    return _serialize(stored)


def _serialize(event: SafetyEvent) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        return value

    result = {key: convert(value) for key, value in asdict(event).items()}
    result["evidence_url"] = f"/api/events/{event.event_id}/evidence"
    result["clip_url"] = (
        None if event.clip_path is None else f"/api/events/{event.event_id}/clip"
    )
    return result


def _evidence_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="evidence not found")
    return candidate


def _video_response(path: Path, range_header: str | None) -> Response:
    """Serve a short local clip, including the byte ranges used by video players."""

    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes"}
    if range_header and range_header.startswith("bytes="):
        try:
            start_text, end_text = range_header[6:].split("-", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="invalid byte range") from exc
        if start < 0 or start >= size or end < start:
            raise HTTPException(status_code=416, detail="invalid byte range")
        end = min(end, size - 1)
        with path.open("rb") as handle:
            handle.seek(start)
            content = handle.read(end - start + 1)
        headers.update(
            {
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(len(content)),
            }
        )
        return Response(content, status_code=206, media_type="video/mp4", headers=headers)
    return Response(path.read_bytes(), media_type="video/mp4", headers=headers)

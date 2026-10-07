"""Shared data types for video ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class CameraState(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """A decoded image and its position in a source."""

    image: Any
    source_timestamp_ms: float
    source_frame_number: int
    loop_count: int


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A decoded frame enriched with camera and capture metadata."""

    camera_id: str
    frame_id: int
    image: Any
    captured_at: datetime
    captured_monotonic: float
    source_timestamp_ms: float
    source_frame_number: int
    loop_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """Point-in-time, thread-safe view of a camera worker."""

    camera_id: str
    state: CameraState
    frames_read: int
    latest_frame_id: int | None
    latest_frame_age_seconds: float | None
    source_fps: float
    effective_reader_fps: float
    loop_count: int
    last_successful_frame_at: datetime | None
    last_error: str | None

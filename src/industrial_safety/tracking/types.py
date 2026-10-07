"""Immutable tracking records and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from industrial_safety.inference.types import Detection, InferenceResult


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class PersonTrack:
    """Anonymous, camera-local visual trajectory at one observation time."""

    camera_id: str
    track_id: int
    frame_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    age_seconds: float
    hits: int
    missed_updates: int
    state: TrackState
    source_detection: Detection | None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.center_x, self.y2)


@dataclass(frozen=True, slots=True)
class TrackingResult:
    camera_id: str
    frame_id: int
    inference_result: InferenceResult
    tracks: tuple[PersonTrack, ...]
    tracking_started_at: datetime
    tracking_finished_at: datetime
    tracking_started_monotonic: float
    tracking_finished_monotonic: float
    tracking_ms: float
    tracker_was_reset: bool
    reset_reason: str | None


@dataclass(frozen=True, slots=True)
class CameraTrackingStats:
    camera_id: str
    latest_inference_frame_id: int | None
    last_tracked_frame_id: int | None
    tracking_updates: int
    tracking_fps: float
    tracking_errors: int
    current_active_tracks: int
    current_confirmed_tracks: int
    current_lost_tracks: int
    total_tracks_created: int
    expired_tracks: int
    tracker_resets: int
    mean_tracking_ms: float
    median_tracking_ms: float
    p95_tracking_ms: float
    max_tracking_ms: float
    inference_result_gaps: int
    maximum_frame_gap: int
    mean_observation_interval_seconds: float
    id_switches: int | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class GlobalTrackingStats:
    tracking_updates: int
    tracking_fps: float
    tracking_errors: int
    mean_tracking_ms: float
    median_tracking_ms: float
    p95_tracking_ms: float
    max_tracking_ms: float

"""Immutable person-detection and inference result records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Detection:
    camera_id: str
    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    detected_at: datetime

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


@dataclass(frozen=True, slots=True)
class InferenceResult:
    camera_id: str
    frame_id: int
    captured_at: datetime
    source_timestamp_ms: float
    source_frame_number: int
    source_loop_count: int
    inference_started_at: datetime
    inference_finished_at: datetime
    inference_started_monotonic: float
    inference_finished_monotonic: float
    detections: tuple[Detection, ...]
    inference_ms: float
    frame_age_at_start_ms: float
    end_to_end_age_ms: float

    @property
    def observed_monotonic(self) -> float:
        """Monotonic time of the source observation reconstructed from age metadata."""

        return self.inference_started_monotonic - self.frame_age_at_start_ms / 1000.0


@dataclass(frozen=True, slots=True)
class CameraInferenceStats:
    camera_id: str
    latest_camera_frame_id: int | None
    last_processed_frame_id: int | None
    frames_skipped: int
    inference_calls: int
    successful_inference_calls: int
    inference_errors: int
    inference_fps: float
    mean_inference_ms: float
    median_inference_ms: float
    p95_inference_ms: float
    mean_frame_age_at_start_ms: float
    mean_end_to_end_age_ms: float
    last_end_to_end_age_ms: float | None
    last_person_count: int
    total_person_detections: int
    max_person_count: int
    last_successful_inference_at: datetime | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class GlobalInferenceStats:
    inference_calls: int
    inference_errors: int
    inference_fps: float
    mean_inference_ms: float

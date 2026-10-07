"""Immutable helmet observation, association, and metric records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from industrial_safety.tracking.types import TrackingResult

BBox = tuple[float, float, float, float]


class PPEType(str, Enum):
    HELMET = "HELMET"


class PPEObservationState(str, Enum):
    OBSERVED = "OBSERVED"
    UNOBSERVED = "UNOBSERVED"


@dataclass(frozen=True, slots=True)
class PPEObservation:
    camera_id: str
    frame_id: int
    ppe_type: PPEType
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    observed_at: datetime

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
    def bbox(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class PPEInferenceResult:
    camera_id: str
    frame_id: int
    captured_at: datetime
    source_timestamp_ms: float
    source_frame_number: int
    source_loop_count: int
    observations: tuple[PPEObservation, ...]
    inference_started_at: datetime
    inference_finished_at: datetime
    inference_started_monotonic: float
    inference_finished_monotonic: float
    inference_ms: float
    frame_age_at_start_ms: float
    end_to_end_age_ms: float


@dataclass(frozen=True, slots=True)
class HeadRegion:
    x1: float
    y1: float
    x2: float
    y2: float

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
    def bbox(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class TrackPPEObservation:
    """Helmet evidence for one currently observed confirmed person track."""

    camera_id: str
    frame_id: int
    track_id: int
    ppe_type: PPEType
    state: PPEObservationState
    head_region: HeadRegion
    observation: PPEObservation | None
    association_score: float | None

    @property
    def confidence(self) -> float | None:
        return None if self.observation is None else self.observation.confidence

    @property
    def ppe_bbox(self) -> BBox | None:
        return None if self.observation is None else self.observation.bbox


@dataclass(frozen=True, slots=True)
class PPEAssociationResult:
    camera_id: str
    frame_id: int
    tracking_result: TrackingResult
    ppe_inference_result: PPEInferenceResult
    track_observations: tuple[TrackPPEObservation, ...]
    unassociated_observations: tuple[PPEObservation, ...]
    association_started_at: datetime
    association_finished_at: datetime
    association_started_monotonic: float
    association_finished_monotonic: float
    association_ms: float


@dataclass(frozen=True, slots=True)
class CameraPPEStats:
    camera_id: str
    latest_tracking_frame_id: int | None
    last_processed_frame_id: int | None
    inference_calls: int
    successful_inference_calls: int
    inference_errors: int
    frame_mismatches: int
    ppe_fps: float
    mean_inference_ms: float
    median_inference_ms: float
    p95_inference_ms: float
    mean_association_ms: float
    median_association_ms: float
    p95_association_ms: float
    mean_end_to_end_age_ms: float
    observed_tracks: int
    unobserved_tracks: int
    unassociated_observations: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class GlobalPPEStats:
    inference_calls: int
    successful_inference_calls: int
    inference_errors: int
    ppe_fps: float
    mean_inference_ms: float
    median_inference_ms: float
    p95_inference_ms: float
    mean_association_ms: float
    median_association_ms: float
    p95_association_ms: float
    mean_end_to_end_age_ms: float

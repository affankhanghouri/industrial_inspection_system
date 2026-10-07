"""Bounded inference metrics aggregation."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from .types import CameraInferenceStats


@dataclass(slots=True)
class MutableCameraInferenceStats:
    camera_id: str
    started_monotonic: float
    last_attempted_frame_id: int | None = None
    last_processed_frame_id: int | None = None
    skipped_before_attempts: int = 0
    inference_calls: int = 0
    inference_errors: int = 0
    successful_calls: int = 0
    total_inference_ms: float = 0.0
    total_frame_age_at_start_ms: float = 0.0
    total_end_to_end_age_ms: float = 0.0
    last_end_to_end_age_ms: float | None = None
    last_person_count: int = 0
    total_person_detections: int = 0
    max_person_count: int = 0
    last_successful_inference_at: datetime | None = None
    last_error: str | None = None
    recent_inference_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    def begin_attempt(self, frame_id: int) -> None:
        previous = self.last_attempted_frame_id
        self.skipped_before_attempts += (
            max(0, frame_id) if previous is None else max(0, frame_id - previous - 1)
        )
        self.last_attempted_frame_id = frame_id
        self.inference_calls += 1

    def record_success(
        self,
        *,
        frame_id: int,
        inference_ms: float,
        frame_age_at_start_ms: float,
        end_to_end_age_ms: float,
        person_count: int,
        finished_at: datetime,
    ) -> None:
        self.last_processed_frame_id = frame_id
        self.successful_calls += 1
        self.total_inference_ms += inference_ms
        self.total_frame_age_at_start_ms += frame_age_at_start_ms
        self.total_end_to_end_age_ms += end_to_end_age_ms
        self.last_end_to_end_age_ms = end_to_end_age_ms
        self.last_person_count = person_count
        self.total_person_detections += person_count
        self.max_person_count = max(self.max_person_count, person_count)
        self.last_successful_inference_at = finished_at
        self.last_error = None
        self.recent_inference_ms.append(inference_ms)

    def record_error(self, message: str) -> None:
        self.inference_errors += 1
        self.last_error = message

    def snapshot(
        self,
        *,
        latest_camera_frame_id: int | None,
        now_monotonic: float,
        stopped_monotonic: float | None,
    ) -> CameraInferenceStats:
        durations = tuple(self.recent_inference_ms)
        elapsed = max(
            0.0, (stopped_monotonic or now_monotonic) - self.started_monotonic
        )
        successful = self.successful_calls
        trailing_skipped = 0
        if latest_camera_frame_id is not None:
            if self.last_attempted_frame_id is None:
                trailing_skipped = max(0, latest_camera_frame_id)
            else:
                trailing_skipped = max(
                    0, latest_camera_frame_id - self.last_attempted_frame_id
                )
        return CameraInferenceStats(
            camera_id=self.camera_id,
            latest_camera_frame_id=latest_camera_frame_id,
            last_processed_frame_id=self.last_processed_frame_id,
            frames_skipped=self.skipped_before_attempts + trailing_skipped,
            inference_calls=self.inference_calls,
            successful_inference_calls=self.successful_calls,
            inference_errors=self.inference_errors,
            inference_fps=successful / elapsed if elapsed > 0 else 0.0,
            mean_inference_ms=(self.total_inference_ms / successful if successful else 0.0),
            median_inference_ms=(statistics.median(durations) if durations else 0.0),
            p95_inference_ms=_percentile(durations, 0.95),
            mean_frame_age_at_start_ms=(
                self.total_frame_age_at_start_ms / successful if successful else 0.0
            ),
            mean_end_to_end_age_ms=(
                self.total_end_to_end_age_ms / successful if successful else 0.0
            ),
            last_end_to_end_age_ms=self.last_end_to_end_age_ms,
            last_person_count=self.last_person_count,
            total_person_detections=self.total_person_detections,
            max_person_count=self.max_person_count,
            last_successful_inference_at=self.last_successful_inference_at,
            last_error=self.last_error,
        )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]

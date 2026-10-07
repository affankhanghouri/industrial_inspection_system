"""Bounded PPE inference and association metrics."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from .types import (
    CameraPPEStats,
    PPEAssociationResult,
    PPEInferenceResult,
    PPEObservationState,
)


@dataclass(slots=True)
class MutableCameraPPEStats:
    camera_id: str
    started_monotonic: float
    last_attempted_frame_id: int | None = None
    last_processed_frame_id: int | None = None
    inference_calls: int = 0
    successful_inference_calls: int = 0
    inference_errors: int = 0
    frame_mismatches: int = 0
    total_inference_ms: float = 0.0
    total_association_ms: float = 0.0
    total_end_to_end_age_ms: float = 0.0
    observed_tracks: int = 0
    unobserved_tracks: int = 0
    unassociated_observations: int = 0
    last_error: str | None = None
    recent_inference_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    recent_association_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    def begin_attempt(self, frame_id: int) -> None:
        self.last_attempted_frame_id = frame_id
        self.inference_calls += 1

    def record_frame_mismatch(self, frame_id: int) -> None:
        self.last_attempted_frame_id = frame_id
        self.frame_mismatches += 1

    def record_success(
        self,
        inference: PPEInferenceResult,
        association: PPEAssociationResult,
    ) -> None:
        self.last_processed_frame_id = inference.frame_id
        self.successful_inference_calls += 1
        self.total_inference_ms += inference.inference_ms
        self.total_association_ms += association.association_ms
        self.total_end_to_end_age_ms += inference.end_to_end_age_ms
        self.observed_tracks += sum(
            item.state is PPEObservationState.OBSERVED
            for item in association.track_observations
        )
        self.unobserved_tracks += sum(
            item.state is PPEObservationState.UNOBSERVED
            for item in association.track_observations
        )
        self.unassociated_observations += len(association.unassociated_observations)
        self.last_error = None
        self.recent_inference_ms.append(inference.inference_ms)
        self.recent_association_ms.append(association.association_ms)

    def record_error(self, message: str) -> None:
        self.inference_errors += 1
        self.last_error = message

    def snapshot(
        self,
        latest_tracking_frame_id: int | None,
        *,
        elapsed_seconds: float,
    ) -> CameraPPEStats:
        inference_durations = tuple(self.recent_inference_ms)
        association_durations = tuple(self.recent_association_ms)
        successful = self.successful_inference_calls
        return CameraPPEStats(
            camera_id=self.camera_id,
            latest_tracking_frame_id=latest_tracking_frame_id,
            last_processed_frame_id=self.last_processed_frame_id,
            inference_calls=self.inference_calls,
            successful_inference_calls=successful,
            inference_errors=self.inference_errors,
            frame_mismatches=self.frame_mismatches,
            ppe_fps=successful / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            mean_inference_ms=(
                self.total_inference_ms / successful if successful else 0.0
            ),
            median_inference_ms=(
                statistics.median(inference_durations)
                if inference_durations
                else 0.0
            ),
            p95_inference_ms=_percentile(inference_durations, 0.95),
            mean_association_ms=(
                self.total_association_ms / successful if successful else 0.0
            ),
            median_association_ms=(
                statistics.median(association_durations)
                if association_durations
                else 0.0
            ),
            p95_association_ms=_percentile(association_durations, 0.95),
            mean_end_to_end_age_ms=(
                self.total_end_to_end_age_ms / successful if successful else 0.0
            ),
            observed_tracks=self.observed_tracks,
            unobserved_tracks=self.unobserved_tracks,
            unassociated_observations=self.unassociated_observations,
            last_error=self.last_error,
        )


def percentile(values: tuple[float, ...], quantile: float) -> float:
    return _percentile(values, quantile)


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]

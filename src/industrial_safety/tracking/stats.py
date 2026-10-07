"""Bounded tracking metrics aggregation."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from .types import CameraTrackingStats, TrackingResult, TrackState


@dataclass(slots=True)
class MutableCameraTrackingStats:
    camera_id: str
    last_attempted_frame_id: int | None = None
    last_tracked_frame_id: int | None = None
    tracking_updates: int = 0
    tracking_errors: int = 0
    total_tracks_created: int = 0
    expired_tracks: int = 0
    tracker_resets: int = 0
    current_active_tracks: int = 0
    current_confirmed_tracks: int = 0
    current_lost_tracks: int = 0
    inference_result_gaps: int = 0
    maximum_frame_gap: int = 0
    total_observation_interval_seconds: float = 0.0
    observation_intervals: int = 0
    last_observed_monotonic: float | None = None
    last_error: str | None = None
    previous_track_ids: set[int] = field(default_factory=set)
    recent_tracking_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    def begin_attempt(self, frame_id: int) -> None:
        self.last_attempted_frame_id = frame_id

    def record_success(self, result: TrackingResult) -> None:
        previous_frame = self.last_tracked_frame_id
        if previous_frame is not None and result.frame_id > previous_frame:
            gap = result.frame_id - previous_frame
            self.inference_result_gaps += max(0, gap - 1)
            self.maximum_frame_gap = max(self.maximum_frame_gap, gap)
        observed = result.inference_result.observed_monotonic
        if self.last_observed_monotonic is not None:
            interval = max(0.0, observed - self.last_observed_monotonic)
            self.total_observation_interval_seconds += interval
            self.observation_intervals += 1
        self.last_observed_monotonic = observed

        current_ids = {track.track_id for track in result.tracks}
        if result.tracker_was_reset:
            self.tracker_resets += 1
            self.previous_track_ids.clear()
        else:
            self.expired_tracks += len(self.previous_track_ids - current_ids)
        self.total_tracks_created += sum(
            1 for track in result.tracks if track.hits == 1 and track.missed_updates == 0
        )
        self.previous_track_ids = current_ids
        self.current_active_tracks = sum(
            track.state is not TrackState.LOST for track in result.tracks
        )
        self.current_confirmed_tracks = sum(
            track.state is TrackState.CONFIRMED for track in result.tracks
        )
        self.current_lost_tracks = sum(
            track.state is TrackState.LOST for track in result.tracks
        )
        self.tracking_updates += 1
        self.last_tracked_frame_id = result.frame_id
        self.last_error = None
        self.recent_tracking_ms.append(result.tracking_ms)

    def record_error(self, message: str, *, tracker_was_reset: bool) -> None:
        self.tracking_errors += 1
        self.last_error = message
        if tracker_was_reset:
            self.tracker_resets += 1
            self.previous_track_ids.clear()
            self.current_active_tracks = 0
            self.current_confirmed_tracks = 0
            self.current_lost_tracks = 0

    def snapshot(
        self,
        latest_inference_frame_id: int | None,
        *,
        elapsed_seconds: float,
    ) -> CameraTrackingStats:
        durations = tuple(self.recent_tracking_ms)
        return CameraTrackingStats(
            camera_id=self.camera_id,
            latest_inference_frame_id=latest_inference_frame_id,
            last_tracked_frame_id=self.last_tracked_frame_id,
            tracking_updates=self.tracking_updates,
            tracking_fps=(
                self.tracking_updates / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
            tracking_errors=self.tracking_errors,
            current_active_tracks=self.current_active_tracks,
            current_confirmed_tracks=self.current_confirmed_tracks,
            current_lost_tracks=self.current_lost_tracks,
            total_tracks_created=self.total_tracks_created,
            expired_tracks=self.expired_tracks,
            tracker_resets=self.tracker_resets,
            mean_tracking_ms=(statistics.mean(durations) if durations else 0.0),
            median_tracking_ms=(statistics.median(durations) if durations else 0.0),
            p95_tracking_ms=_percentile(durations, 0.95),
            max_tracking_ms=(max(durations) if durations else 0.0),
            inference_result_gaps=self.inference_result_gaps,
            maximum_frame_gap=self.maximum_frame_gap,
            mean_observation_interval_seconds=(
                self.total_observation_interval_seconds / self.observation_intervals
                if self.observation_intervals
                else 0.0
            ),
            id_switches=None,
            last_error=self.last_error,
        )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]

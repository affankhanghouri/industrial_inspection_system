"""Per-camera tracking processors and latest-result coordinator."""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from industrial_safety.inference.latest_result import LatestInferenceResult
from industrial_safety.inference.types import InferenceResult

from .latest_result import LatestTrackingResult
from .stats import MutableCameraTrackingStats, _percentile
from .tracker import PersonTracker
from .types import CameraTrackingStats, GlobalTrackingStats, TrackingResult

TrackerFactory = Callable[[str], PersonTracker]


class CameraTrackingProcessor:
    """Own discontinuity detection and exactly one camera-local tracker."""

    def __init__(self, camera_id: str, tracker: PersonTracker) -> None:
        if tracker.camera_id != camera_id:
            raise ValueError("tracker camera ID does not match processor camera ID")
        self.camera_id = camera_id
        self.tracker = tracker
        self._last_frame_id: int | None = None
        self._last_source_loop_count: int | None = None
        self._last_source_timestamp_ms: float | None = None

    def process(self, inference: InferenceResult) -> TrackingResult:
        if inference.camera_id != self.camera_id:
            raise ValueError(
                f"processor {self.camera_id} cannot process {inference.camera_id}"
            )
        reset_reason = self._discontinuity_reason(inference)
        if reset_reason is not None:
            self.tracker.reset()

        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        tracks = self.tracker.update(inference)
        finished_monotonic = time.monotonic()
        finished_at = datetime.now(timezone.utc)
        self._last_frame_id = inference.frame_id
        self._last_source_loop_count = inference.source_loop_count
        self._last_source_timestamp_ms = inference.source_timestamp_ms
        return TrackingResult(
            camera_id=self.camera_id,
            frame_id=inference.frame_id,
            inference_result=inference,
            tracks=tracks,
            tracking_started_at=started_at,
            tracking_finished_at=finished_at,
            tracking_started_monotonic=started_monotonic,
            tracking_finished_monotonic=finished_monotonic,
            tracking_ms=max(0.0, (finished_monotonic - started_monotonic) * 1000.0),
            tracker_was_reset=reset_reason is not None,
            reset_reason=reset_reason,
        )

    def reset(self) -> None:
        self.tracker.reset()
        self._last_frame_id = None
        self._last_source_loop_count = None
        self._last_source_timestamp_ms = None

    def close(self) -> None:
        self.tracker.close()

    def _discontinuity_reason(self, inference: InferenceResult) -> str | None:
        if (
            self._last_source_loop_count is not None
            and inference.source_loop_count != self._last_source_loop_count
        ):
            return (
                f"source loop changed {self._last_source_loop_count}"
                f"->{inference.source_loop_count}"
            )
        if self._last_frame_id is not None and inference.frame_id <= self._last_frame_id:
            return f"frame ID regressed {self._last_frame_id}->{inference.frame_id}"
        if (
            self._last_source_timestamp_ms is not None
            and inference.source_timestamp_ms + 1.0 < self._last_source_timestamp_ms
        ):
            return (
                f"source timestamp regressed {self._last_source_timestamp_ms:.1f}"
                f"->{inference.source_timestamp_ms:.1f}ms"
            )
        return None


class TrackingCoordinator:
    """Consume only latest inference results and update isolated camera trackers."""

    def __init__(
        self,
        inference_results: dict[str, LatestInferenceResult],
        *,
        tracker_factory: TrackerFactory | None = None,
        processors: dict[str, CameraTrackingProcessor] | None = None,
        idle_wait_seconds: float = 0.005,
        logger: logging.Logger | None = None,
    ) -> None:
        if not inference_results:
            raise ValueError("tracking coordinator requires at least one camera")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be greater than zero")
        self._inference_results = dict(inference_results)
        self._camera_ids = tuple(inference_results)
        if processors is None:
            if tracker_factory is None:
                raise ValueError("tracker_factory or processors is required")
            processors = {
                camera_id: CameraTrackingProcessor(
                    camera_id, tracker_factory(camera_id)
                )
                for camera_id in self._camera_ids
            }
        if set(processors) != set(self._camera_ids):
            raise ValueError("processors must exactly match inference cameras")
        self._processors = dict(processors)
        self._idle_wait_seconds = idle_wait_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._results = {
            camera_id: LatestTrackingResult() for camera_id in self._camera_ids
        }
        self._stats_lock = threading.Lock()
        self._stats: dict[str, MutableCameraTrackingStats] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None
        self._stopped_monotonic: float | None = None
        self._fatal_error: str | None = None

    @property
    def results(self) -> dict[str, LatestTrackingResult]:
        return dict(self._results)

    @property
    def processors(self) -> dict[str, CameraTrackingProcessor]:
        return dict(self._processors)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fatal_error(self) -> str | None:
        with self._stats_lock:
            return self._fatal_error

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        started = time.monotonic()
        with self._stats_lock:
            self._started_monotonic = started
            self._stopped_monotonic = None
            self._fatal_error = None
            self._stats = {
                camera_id: MutableCameraTrackingStats(camera_id)
                for camera_id in self._camera_ids
            }
        for processor in self._processors.values():
            processor.reset()
        self._thread = threading.Thread(
            target=self._run,
            name="person-tracking-coordinator",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def camera_stats(self) -> dict[str, CameraTrackingStats]:
        snapshots: dict[str, CameraTrackingStats] = {}
        now = time.monotonic()
        with self._stats_lock:
            started = self._started_monotonic
            end = self._stopped_monotonic or now
            elapsed = 0.0 if started is None else max(0.0, end - started)
            for camera_id, mutable in self._stats.items():
                latest = self._inference_results[camera_id].get_latest(timeout=0.0)
                latest_id = None if latest is None else latest.frame_id
                snapshots[camera_id] = mutable.snapshot(
                    latest_id, elapsed_seconds=elapsed
                )
        return snapshots

    def global_stats(self) -> GlobalTrackingStats:
        camera_stats = self.camera_stats().values()
        updates = sum(stats.tracking_updates for stats in camera_stats)
        errors = sum(stats.tracking_errors for stats in camera_stats)
        durations: list[float] = []
        with self._stats_lock:
            for mutable in self._stats.values():
                durations.extend(mutable.recent_tracking_ms)
            started = self._started_monotonic
            end = self._stopped_monotonic or time.monotonic()
            elapsed = 0.0 if started is None else max(0.0, end - started)
        values = tuple(durations)
        return GlobalTrackingStats(
            tracking_updates=updates,
            tracking_fps=updates / elapsed if elapsed > 0 else 0.0,
            tracking_errors=errors,
            mean_tracking_ms=(statistics.mean(values) if values else 0.0),
            median_tracking_ms=(statistics.median(values) if values else 0.0),
            p95_tracking_ms=_percentile(values, 0.95),
            max_tracking_ms=(max(values) if values else 0.0),
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                processed_any = False
                for camera_id in self._camera_ids:
                    if self._stop_event.is_set():
                        break
                    inference = self._inference_results[camera_id].get_latest(
                        timeout=0.0
                    )
                    if inference is None:
                        continue
                    with self._stats_lock:
                        last_attempted = self._stats[camera_id].last_attempted_frame_id
                        if inference.frame_id == last_attempted:
                            continue
                        self._stats[camera_id].begin_attempt(inference.frame_id)
                    processed_any = True
                    self._track(camera_id, inference)
                if not processed_any:
                    self._stop_event.wait(self._idle_wait_seconds)
        except Exception as exc:
            with self._stats_lock:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("tracking coordinator failed")
        finally:
            with self._stats_lock:
                self._stopped_monotonic = time.monotonic()
            for processor in self._processors.values():
                try:
                    processor.close()
                except Exception:
                    self._logger.exception("tracker close failed")
            for result_buffer in self._results.values():
                result_buffer.close()
            self._logger.info("tracking coordinator stopped")

    def _track(self, camera_id: str, inference: InferenceResult) -> None:
        try:
            result = self._processors[camera_id].process(inference)
            if result.tracker_was_reset:
                self._logger.info(
                    "[%s] tracker reset: %s", camera_id, result.reset_reason
                )
            self._results[camera_id].publish(result)
            with self._stats_lock:
                self._stats[camera_id].record_success(result)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            reset_succeeded = False
            try:
                self._processors[camera_id].reset()
                reset_succeeded = True
            except Exception:
                self._logger.exception("[%s] tracker reset after error failed", camera_id)
            with self._stats_lock:
                self._stats[camera_id].record_error(
                    message, tracker_was_reset=reset_succeeded
                )
            self._logger.exception(
                "[%s] tracking failed for frame %d", camera_id, inference.frame_id
            )

"""Latest-only, camera-selective PPE inference and exact-frame association."""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from industrial_safety.tracking.latest_result import LatestTrackingResult
from industrial_safety.tracking.types import TrackingResult
from industrial_safety.video.types import FramePacket

from .association import HelmetTrackAssociator
from .detector import PPEDetector
from .latest_result import LatestPPEAssociationResult, LatestPPEInferenceResult
from .stats import MutableCameraPPEStats, percentile
from .types import CameraPPEStats, GlobalPPEStats, PPEInferenceResult

FrameProvider = Callable[[str], FramePacket | None]


class PPECoordinator:
    """Share one PPE detector across explicitly enabled camera tracks."""

    def __init__(
        self,
        tracking_results: dict[str, LatestTrackingResult],
        frame_provider: FrameProvider,
        detector: PPEDetector,
        associator: HelmetTrackAssociator,
        *,
        enabled_camera_ids: tuple[str, ...],
        minimum_interval_seconds: float = 0.0,
        idle_wait_seconds: float = 0.005,
        retain_processed_frames: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if not enabled_camera_ids:
            raise ValueError("PPE requires at least one enabled camera")
        if not set(enabled_camera_ids).issubset(tracking_results):
            raise ValueError("PPE enabled cameras must exist in tracking results")
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be greater than zero")
        self._tracking_results = dict(tracking_results)
        self._frame_provider = frame_provider
        self._detector = detector
        self._associator = associator
        self._camera_ids = tuple(enabled_camera_ids)
        self._minimum_interval_seconds = minimum_interval_seconds
        self._idle_wait_seconds = idle_wait_seconds
        self._retain_processed_frames = retain_processed_frames
        self._logger = logger or logging.getLogger(__name__)
        self._inference_results = {
            camera_id: LatestPPEInferenceResult() for camera_id in self._camera_ids
        }
        self._association_results = {
            camera_id: LatestPPEAssociationResult() for camera_id in self._camera_ids
        }
        self._stats_lock = threading.Lock()
        self._stats: dict[str, MutableCameraPPEStats] = {}
        self._processed_frames: dict[str, FramePacket] = {}
        self._last_started: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None
        self._stopped_monotonic: float | None = None
        self._fatal_error: str | None = None

    @property
    def inference_results(self) -> dict[str, LatestPPEInferenceResult]:
        return dict(self._inference_results)

    @property
    def association_results(self) -> dict[str, LatestPPEAssociationResult]:
        return dict(self._association_results)

    @property
    def enabled_camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fatal_error(self) -> str | None:
        with self._stats_lock:
            return self._fatal_error

    def latest_processed_frame(self, camera_id: str) -> FramePacket | None:
        with self._stats_lock:
            return self._processed_frames.get(camera_id)

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        started = time.monotonic()
        with self._stats_lock:
            self._started_monotonic = started
            self._stopped_monotonic = None
            self._fatal_error = None
            self._processed_frames.clear()
            self._last_started.clear()
            self._stats = {
                camera_id: MutableCameraPPEStats(camera_id, started)
                for camera_id in self._camera_ids
            }
        self._thread = threading.Thread(
            target=self._run,
            name="helmet-ppe-coordinator",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def camera_stats(self) -> dict[str, CameraPPEStats]:
        now = time.monotonic()
        snapshots: dict[str, CameraPPEStats] = {}
        with self._stats_lock:
            started = self._started_monotonic
            end = self._stopped_monotonic or now
            elapsed = 0.0 if started is None else max(0.0, end - started)
            for camera_id, mutable in self._stats.items():
                tracking = self._tracking_results[camera_id].get_latest(timeout=0.0)
                latest_id = None if tracking is None else tracking.frame_id
                snapshots[camera_id] = mutable.snapshot(
                    latest_id, elapsed_seconds=elapsed
                )
        return snapshots

    def global_stats(self) -> GlobalPPEStats:
        camera_stats = self.camera_stats().values()
        calls = sum(item.inference_calls for item in camera_stats)
        successful = sum(item.successful_inference_calls for item in camera_stats)
        errors = sum(item.inference_errors for item in camera_stats)
        with self._stats_lock:
            inference_values = tuple(
                value
                for mutable in self._stats.values()
                for value in mutable.recent_inference_ms
            )
            association_values = tuple(
                value
                for mutable in self._stats.values()
                for value in mutable.recent_association_ms
            )
            total_age = sum(
                mutable.total_end_to_end_age_ms for mutable in self._stats.values()
            )
            started = self._started_monotonic
            end = self._stopped_monotonic or time.monotonic()
            elapsed = 0.0 if started is None else max(0.0, end - started)
        return GlobalPPEStats(
            inference_calls=calls,
            successful_inference_calls=successful,
            inference_errors=errors,
            ppe_fps=successful / elapsed if elapsed > 0 else 0.0,
            mean_inference_ms=(statistics.mean(inference_values) if inference_values else 0.0),
            median_inference_ms=(
                statistics.median(inference_values) if inference_values else 0.0
            ),
            p95_inference_ms=percentile(inference_values, 0.95),
            mean_association_ms=(
                statistics.mean(association_values) if association_values else 0.0
            ),
            median_association_ms=(
                statistics.median(association_values)
                if association_values
                else 0.0
            ),
            p95_association_ms=percentile(association_values, 0.95),
            mean_end_to_end_age_ms=total_age / successful if successful else 0.0,
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                processed_any = False
                for camera_id in self._camera_ids:
                    if self._stop_event.is_set():
                        break
                    tracking = self._tracking_results[camera_id].get_latest(timeout=0.0)
                    if tracking is None:
                        continue
                    with self._stats_lock:
                        last_attempted = self._stats[camera_id].last_attempted_frame_id
                        last_started = self._last_started.get(camera_id)
                    if tracking.frame_id == last_attempted:
                        continue
                    now = time.monotonic()
                    if (
                        last_started is not None
                        and now - last_started < self._minimum_interval_seconds
                    ):
                        continue
                    packet = self._frame_provider(camera_id)
                    if packet is None or packet.frame_id < tracking.frame_id:
                        # The inference scheduler may still be publishing the exact
                        # processed frame. Do not discard the tracking result yet.
                        continue
                    if packet.frame_id > tracking.frame_id:
                        with self._stats_lock:
                            self._stats[camera_id].record_frame_mismatch(
                                tracking.frame_id
                            )
                        continue
                    with self._stats_lock:
                        self._stats[camera_id].begin_attempt(tracking.frame_id)
                        self._last_started[camera_id] = now
                    processed_any = True
                    self._process(camera_id, packet, tracking)
                if not processed_any:
                    self._stop_event.wait(self._idle_wait_seconds)
        except Exception as exc:
            with self._stats_lock:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("PPE coordinator failed")
        finally:
            with self._stats_lock:
                self._stopped_monotonic = time.monotonic()
            try:
                self._detector.close()
            except Exception:
                self._logger.exception("PPE detector close failed")
            for result in self._inference_results.values():
                result.close()
            for result in self._association_results.values():
                result.close()
            self._logger.info("PPE coordinator stopped")

    def _process(
        self,
        camera_id: str,
        packet: FramePacket,
        tracking: TrackingResult,
    ) -> None:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        frame_age_at_start_ms = max(
            0.0, (started_monotonic - packet.captured_monotonic) * 1000.0
        )
        try:
            observations = self._detector.detect(packet)
            if not isinstance(observations, tuple):
                raise TypeError("PPE detector must return tuple[PPEObservation, ...]")
            if any(
                item.camera_id != camera_id or item.frame_id != packet.frame_id
                for item in observations
            ):
                raise ValueError("PPE detector returned mismatched observation identity")
            finished_monotonic = time.monotonic()
            finished_at = datetime.now(timezone.utc)
            inference = PPEInferenceResult(
                camera_id=camera_id,
                frame_id=packet.frame_id,
                captured_at=packet.captured_at,
                source_timestamp_ms=packet.source_timestamp_ms,
                source_frame_number=packet.source_frame_number,
                source_loop_count=packet.loop_count,
                observations=observations,
                inference_started_at=started_at,
                inference_finished_at=finished_at,
                inference_started_monotonic=started_monotonic,
                inference_finished_monotonic=finished_monotonic,
                inference_ms=max(
                    0.0, (finished_monotonic - started_monotonic) * 1000.0
                ),
                frame_age_at_start_ms=frame_age_at_start_ms,
                end_to_end_age_ms=max(
                    0.0, (finished_monotonic - packet.captured_monotonic) * 1000.0
                ),
            )
            self._inference_results[camera_id].publish(inference)
            association = self._associator.associate(tracking, inference)
            self._association_results[camera_id].publish(association)
            with self._stats_lock:
                self._stats[camera_id].record_success(inference, association)
                if self._retain_processed_frames:
                    self._processed_frames[camera_id] = packet
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._stats_lock:
                self._stats[camera_id].record_error(message)
            self._logger.exception(
                "[%s] PPE processing failed for frame %d",
                camera_id,
                packet.frame_id,
            )

"""Fair, latest-frame-only inference scheduling."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Protocol

from industrial_safety.video.latest_frame import LatestFrameBuffer
from industrial_safety.video.types import FramePacket

from .detector import Detector
from .latest_result import LatestInferenceResult
from .stats import MutableCameraInferenceStats
from .types import CameraInferenceStats, GlobalInferenceStats, InferenceResult


class FrameProvider(Protocol):
    latest_frame: LatestFrameBuffer


class RoundRobinInferenceScheduler:
    """Service each camera in turn using one shared detector instance."""

    def __init__(
        self,
        providers: dict[str, FrameProvider],
        detector: Detector,
        *,
        idle_wait_seconds: float = 0.01,
        retain_processed_frames: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if not providers:
            raise ValueError("inference scheduler requires at least one camera")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be greater than zero")
        self._providers = dict(providers)
        self._camera_ids = tuple(providers)
        self._detector = detector
        self._idle_wait_seconds = idle_wait_seconds
        self._retain_processed_frames = retain_processed_frames
        self._logger = logger or logging.getLogger(__name__)
        self._results = {
            camera_id: LatestInferenceResult() for camera_id in self._camera_ids
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, MutableCameraInferenceStats] = {}
        self._processed_frames: dict[str, FramePacket] = {}
        self._started_monotonic: float | None = None
        self._stopped_monotonic: float | None = None
        self._fatal_error: str | None = None

    @property
    def results(self) -> dict[str, LatestInferenceResult]:
        return dict(self._results)

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
            self._processed_frames.clear()
            self._stats = {
                camera_id: MutableCameraInferenceStats(camera_id, started)
                for camera_id in self._camera_ids
            }
        self._thread = threading.Thread(
            target=self._run,
            name="person-inference-scheduler",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def latest_processed_frame(self, camera_id: str) -> FramePacket | None:
        with self._stats_lock:
            return self._processed_frames.get(camera_id)

    def camera_stats(self) -> dict[str, CameraInferenceStats]:
        now = time.monotonic()
        snapshots: dict[str, CameraInferenceStats] = {}
        with self._stats_lock:
            stopped = self._stopped_monotonic
            for camera_id, mutable in self._stats.items():
                packet = self._providers[camera_id].latest_frame.get_latest(timeout=0.0)
                latest_id = None if packet is None else packet.frame_id
                snapshots[camera_id] = mutable.snapshot(
                    latest_camera_frame_id=latest_id,
                    now_monotonic=now,
                    stopped_monotonic=stopped,
                )
        return snapshots

    def global_stats(self) -> GlobalInferenceStats:
        camera_stats = self.camera_stats()
        calls = sum(stats.inference_calls for stats in camera_stats.values())
        errors = sum(stats.inference_errors for stats in camera_stats.values())
        successful = sum(
            stats.successful_inference_calls for stats in camera_stats.values()
        )
        weighted_latency = sum(
            stats.mean_inference_ms * stats.successful_inference_calls
            for stats in camera_stats.values()
        )
        started = self._started_monotonic
        end = self._stopped_monotonic or time.monotonic()
        elapsed = 0.0 if started is None else max(0.0, end - started)
        return GlobalInferenceStats(
            inference_calls=calls,
            inference_errors=errors,
            inference_fps=successful / elapsed if elapsed > 0 else 0.0,
            mean_inference_ms=weighted_latency / successful if successful else 0.0,
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                processed_any = False
                for camera_id in self._camera_ids:
                    if self._stop_event.is_set():
                        break
                    with self._stats_lock:
                        last_attempted = self._stats[camera_id].last_attempted_frame_id
                    # Re-read at the actual turn; no earlier frame is reserved or queued.
                    packet = self._providers[camera_id].latest_frame.get_latest(
                        after_frame_id=last_attempted,
                        timeout=0.0,
                    )
                    if packet is None:
                        continue
                    processed_any = True
                    self._infer(camera_id, packet)
                if not processed_any:
                    self._stop_event.wait(self._idle_wait_seconds)
        except Exception as exc:
            with self._stats_lock:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("inference scheduler failed")
        finally:
            with self._stats_lock:
                self._stopped_monotonic = time.monotonic()
            for result_buffer in self._results.values():
                result_buffer.close()
            self._logger.info("inference scheduler stopped")

    def _infer(self, camera_id: str, packet: FramePacket) -> None:
        with self._stats_lock:
            self._stats[camera_id].begin_attempt(packet.frame_id)
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        frame_age_at_start_ms = max(
            0.0, (started_monotonic - packet.captured_monotonic) * 1000.0
        )
        try:
            detections = self._detector.detect(packet)
            if not isinstance(detections, tuple):
                raise TypeError("detector must return tuple[Detection, ...]")
            finished_monotonic = time.monotonic()
            finished_at = datetime.now(timezone.utc)
            inference_ms = max(0.0, (finished_monotonic - started_monotonic) * 1000.0)
            end_to_end_age_ms = max(
                0.0, (finished_monotonic - packet.captured_monotonic) * 1000.0
            )
            result = InferenceResult(
                camera_id=camera_id,
                frame_id=packet.frame_id,
                captured_at=packet.captured_at,
                source_timestamp_ms=packet.source_timestamp_ms,
                source_frame_number=packet.source_frame_number,
                source_loop_count=packet.loop_count,
                inference_started_at=started_at,
                inference_finished_at=finished_at,
                inference_started_monotonic=started_monotonic,
                inference_finished_monotonic=finished_monotonic,
                detections=detections,
                inference_ms=inference_ms,
                frame_age_at_start_ms=frame_age_at_start_ms,
                end_to_end_age_ms=end_to_end_age_ms,
            )
            self._results[camera_id].publish(result)
            with self._stats_lock:
                self._stats[camera_id].record_success(
                    frame_id=packet.frame_id,
                    inference_ms=inference_ms,
                    frame_age_at_start_ms=frame_age_at_start_ms,
                    end_to_end_age_ms=end_to_end_age_ms,
                    person_count=len(detections),
                    finished_at=finished_at,
                )
                if self._retain_processed_frames:
                    self._processed_frames[camera_id] = packet
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._stats_lock:
                self._stats[camera_id].record_error(message)
            self._logger.exception(
                "[%s] inference failed for frame %d", camera_id, packet.frame_id
            )

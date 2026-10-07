"""Per-camera reader runtime."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from .config import CameraConfig
from .latest_frame import LatestFrameBuffer
from .source import VideoSource
from .types import CameraHealth, CameraState, FramePacket


class CameraWorker:
    """Read one source in real time and expose only its newest frame."""

    def __init__(
        self,
        config: CameraConfig,
        source: VideoSource,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.latest_frame = LatestFrameBuffer()
        self._logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._health_lock = threading.Lock()
        self._state = CameraState.STOPPED
        self._frames_read = 0
        self._latest_frame_id: int | None = None
        self._latest_frame_monotonic: float | None = None
        self._last_successful_frame_at: datetime | None = None
        self._last_error: str | None = None
        self._reader_started_monotonic: float | None = None
        self._reader_stopped_monotonic: float | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        with self._health_lock:
            self._state = CameraState.STARTING
            self._frames_read = 0
            self._latest_frame_id = None
            self._latest_frame_monotonic = None
            self._last_successful_frame_at = None
            self._last_error = None
            self._reader_started_monotonic = None
            self._reader_stopped_monotonic = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.config.camera_id}",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def health(self) -> CameraHealth:
        now = time.monotonic()
        with self._health_lock:
            age = (
                None
                if self._latest_frame_monotonic is None
                else max(0.0, now - self._latest_frame_monotonic)
            )
            measurement_end = self._reader_stopped_monotonic or now
            elapsed = (
                0.0
                if self._reader_started_monotonic is None
                else max(0.0, measurement_end - self._reader_started_monotonic)
            )
            effective_fps = self._frames_read / elapsed if elapsed > 0 else 0.0
            return CameraHealth(
                camera_id=self.config.camera_id,
                state=self._state,
                frames_read=self._frames_read,
                latest_frame_id=self._latest_frame_id,
                latest_frame_age_seconds=age,
                source_fps=self.source.fps,
                effective_reader_fps=effective_fps,
                loop_count=self.source.loop_count,
                last_successful_frame_at=self._last_successful_frame_at,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        camera_id = self.config.camera_id
        try:
            self.source.open()
            if self.source.fps <= 0:
                raise RuntimeError(f"invalid source FPS: {self.source.fps}")
            with self._health_lock:
                self._state = CameraState.RUNNING
                self._reader_started_monotonic = time.monotonic()
            self._logger.info(
                "[%s] selected source: %s | name=%s | %dx%d @ %.3f FPS",
                camera_id,
                self.config.source_path,
                self.config.name,
                self.source.width,
                self.source.height,
                self.source.fps,
            )

            frame_period = 1.0 / self.source.fps
            next_frame_at = time.monotonic()
            frame_id = 0
            previous_loop_count = self.source.loop_count

            while not self._stop_event.is_set():
                delay = next_frame_at - time.monotonic()
                if delay > 0 and self._stop_event.wait(delay):
                    break

                source_frame = self.source.read()
                captured_monotonic = time.monotonic()
                captured_at = datetime.now(timezone.utc)
                packet = FramePacket(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    image=source_frame.image,
                    captured_at=captured_at,
                    captured_monotonic=captured_monotonic,
                    source_timestamp_ms=source_frame.source_timestamp_ms,
                    source_frame_number=source_frame.source_frame_number,
                    loop_count=source_frame.loop_count,
                    width=self.source.width,
                    height=self.source.height,
                )
                self.latest_frame.publish(packet)
                with self._health_lock:
                    self._frames_read += 1
                    self._latest_frame_id = frame_id
                    self._latest_frame_monotonic = captured_monotonic
                    self._last_successful_frame_at = captured_at

                if source_frame.loop_count != previous_loop_count:
                    previous_loop_count = source_frame.loop_count
                    self._logger.info(
                        "[%s] source looped; loop_count=%d",
                        camera_id,
                        source_frame.loop_count,
                    )
                frame_id += 1
                next_frame_at += frame_period

                # Do not preserve an arbitrarily large timing debt after a decode stall.
                now = time.monotonic()
                if now - next_frame_at > frame_period:
                    next_frame_at = now
        except Exception as exc:
            with self._health_lock:
                self._state = CameraState.ERROR
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("[%s] camera worker failed", camera_id)
        finally:
            try:
                self.source.release()
            except Exception:
                self._logger.exception("[%s] source release failed", camera_id)
            self.latest_frame.close()
            with self._health_lock:
                self._reader_stopped_monotonic = time.monotonic()
                if self._state is not CameraState.ERROR:
                    self._state = CameraState.STOPPED
            self._logger.info("[%s] camera worker stopped", camera_id)

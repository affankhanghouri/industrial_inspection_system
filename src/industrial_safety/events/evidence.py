"""Bounded pre/post-event video capture and asynchronous MP4 writing."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from industrial_safety.video.types import FramePacket

FrameProvider = Callable[[str], FramePacket | None]


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    fps: float = 10.0
    pre_seconds: float = 2.5
    post_seconds: float = 2.5
    max_width: int = 640
    writer_queue_size: int = 8

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.pre_seconds < 0 or self.post_seconds < 0:
            raise ValueError("evidence timing values must be positive")
        if self.max_width <= 0 or self.writer_queue_size <= 0:
            raise ValueError("evidence limits must be positive")

    @property
    def buffer_frames(self) -> int:
        return max(1, math.ceil(self.fps * self.pre_seconds))


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    frame_id: int
    captured_monotonic: float
    image: np.ndarray


class EvidenceRingBuffer:
    """A fixed-size buffer containing only the newest evidence samples."""

    def __init__(self, maximum_frames: int) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self.maximum_frames = maximum_frames
        self._frames: deque[EvidenceFrame] = deque(maxlen=maximum_frames)

    def append(self, sample: EvidenceFrame) -> None:
        self._frames.append(sample)

    def snapshot(self) -> tuple[EvidenceFrame, ...]:
        return tuple(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


@dataclass(slots=True)
class _Capture:
    event_id: str
    camera_id: str
    destination: Path
    deadline_monotonic: float
    frames: list[EvidenceFrame] = field(default_factory=list)
    last_frame_id: int | None = None


class EvidenceRecorder:
    """Sample camera previews into bounded buffers and write event clips off-thread."""

    def __init__(
        self,
        camera_ids: Iterable[str],
        frame_provider: FrameProvider,
        evidence_root: str | Path,
        *,
        config: EvidenceConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or EvidenceConfig()
        self.evidence_root = Path(evidence_root).resolve()
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self._provider = frame_provider
        self._logger = logger or logging.getLogger(__name__)
        self._buffers = {
            camera_id: EvidenceRingBuffer(self.config.buffer_frames)
            for camera_id in camera_ids
        }
        self._last_frame_ids: dict[str, int] = {}
        self._active: dict[str, _Capture] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sample_thread: threading.Thread | None = None
        self._write_queue: queue.Queue[_Capture | None] = queue.Queue(
            maxsize=self.config.writer_queue_size
        )
        self._writer_thread: threading.Thread | None = None
        self._clips_written = 0
        self._write_errors = 0

    @property
    def is_alive(self) -> bool:
        return bool(
            (self._sample_thread and self._sample_thread.is_alive())
            or (self._writer_thread and self._writer_thread.is_alive())
        )

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._write_loop, name="evidence-writer", daemon=False
        )
        self._sample_thread = threading.Thread(
            target=self._sample_loop, name="evidence-sampler", daemon=False
        )
        self._writer_thread.start()
        self._sample_thread.start()

    def stop(self, *, flush: bool = True) -> None:
        self._stop_event.set()
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=5.0)
        if flush:
            with self._lock:
                captures = tuple(self._active.values())
                self._active.clear()
            for capture in captures:
                self._enqueue(capture)
            self._write_queue.join()
        self._write_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)

    def trigger(
        self,
        event_id: str,
        camera_id: str,
        relative_clip_path: str | Path,
        *,
        triggered_monotonic: float | None = None,
    ) -> bool:
        """Start one capture. Repeated triggers for an event are ignored."""

        now = time.monotonic() if triggered_monotonic is None else triggered_monotonic
        with self._lock:
            if camera_id not in self._buffers or event_id in self._active:
                return False
            frames = list(self._buffers[camera_id].snapshot())
            self._active[event_id] = _Capture(
                event_id=event_id,
                camera_id=camera_id,
                destination=(self.evidence_root / relative_clip_path).resolve(),
                deadline_monotonic=now + self.config.post_seconds,
                frames=frames,
                last_frame_id=None if not frames else frames[-1].frame_id,
            )
        return True

    def record_frame(self, packet: FramePacket) -> bool:
        """Record a sampled frame; exposed for deterministic focused tests."""

        with self._lock:
            if packet.camera_id not in self._buffers:
                return False
            if self._last_frame_ids.get(packet.camera_id) == packet.frame_id:
                return False
            sample = EvidenceFrame(
                frame_id=packet.frame_id,
                captured_monotonic=packet.captured_monotonic,
                image=self._prepare_image(packet.image),
            )
            self._last_frame_ids[packet.camera_id] = packet.frame_id
            self._buffers[packet.camera_id].append(sample)
            completed: list[_Capture] = []
            for event_id, capture in tuple(self._active.items()):
                if capture.camera_id != packet.camera_id:
                    continue
                if capture.last_frame_id != packet.frame_id:
                    capture.frames.append(sample)
                    capture.last_frame_id = packet.frame_id
                if packet.captured_monotonic >= capture.deadline_monotonic:
                    completed.append(self._active.pop(event_id))
        for capture in completed:
            self._enqueue(capture)
        return True

    def buffer_size(self, camera_id: str) -> int:
        with self._lock:
            return len(self._buffers[camera_id])

    def pending_event_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._active)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "clips_written": self._clips_written,
                "write_errors": self._write_errors,
                "active_captures": len(self._active),
                "queued_writes": self._write_queue.qsize(),
            }

    def _sample_loop(self) -> None:
        period = 1.0 / self.config.fps
        next_sample = time.monotonic()
        while not self._stop_event.is_set():
            for camera_id in self._buffers:
                packet = self._provider(camera_id)
                if packet is not None:
                    self.record_frame(packet)
            next_sample += period
            delay = next_sample - time.monotonic()
            if delay < -period:
                next_sample = time.monotonic()
                delay = 0.0
            self._stop_event.wait(max(0.0, delay))

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if width <= self.config.max_width:
            return image.copy()
        scale = self.config.max_width / width
        return cv2.resize(
            image,
            (self.config.max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _enqueue(self, capture: _Capture) -> None:
        if not capture.frames:
            self._logger.error("[%s] no frames available for evidence clip", capture.camera_id)
            with self._lock:
                self._write_errors += 1
            return
        try:
            self._write_queue.put(capture, timeout=1.0)
        except queue.Full:
            self._logger.error("evidence writer queue is full; clip %s not written", capture.event_id)
            with self._lock:
                self._write_errors += 1

    def _write_loop(self) -> None:
        while True:
            capture = self._write_queue.get()
            try:
                if capture is None:
                    return
                self._write_clip(capture)
                with self._lock:
                    self._clips_written += 1
            except Exception:
                with self._lock:
                    self._write_errors += 1
                self._logger.exception("failed to write evidence clip")
            finally:
                self._write_queue.task_done()

    def _write_clip(self, capture: _Capture) -> None:
        capture.destination.parent.mkdir(parents=True, exist_ok=True)
        first = capture.frames[0].image
        source_height, source_width = first.shape[:2]
        width = min(source_width, self.config.max_width)
        height = round(source_height * width / source_width)
        writer = cv2.VideoWriter(
            str(capture.destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.config.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open MP4 writer: {capture.destination}")
        try:
            for sample in capture.frames:
                frame = sample.image
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        self._logger.info(
            "[%s] evidence clip: %s (%d frames, %.1fs)",
            capture.camera_id,
            capture.destination,
            len(capture.frames),
            len(capture.frames) / self.config.fps,
        )

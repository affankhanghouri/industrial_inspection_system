from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np

from industrial_safety.inference import Detection, Detector
from industrial_safety.video import FramePacket, LatestFrameBuffer


def frame(camera_id: str, frame_id: int) -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        captured_at=datetime.now(timezone.utc),
        captured_monotonic=time.monotonic(),
        source_timestamp_ms=frame_id * 10.0,
        source_frame_number=frame_id,
        loop_count=0,
        width=16,
        height=16,
    )


def person_detection(packet: FramePacket) -> Detection:
    return Detection(
        camera_id=packet.camera_id,
        frame_id=packet.frame_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=1.0,
        y1=2.0,
        x2=10.0,
        y2=14.0,
        detected_at=datetime.now(timezone.utc),
    )


class FakeProvider:
    def __init__(self) -> None:
        self.latest_frame = LatestFrameBuffer()


class RecordingDetector(Detector):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.called = threading.Event()
        self.block_camera: str | None = None
        self.entered_block = threading.Event()
        self.release_block = threading.Event()
        self.fail_once_for: tuple[str, int] | None = None

    def detect(self, packet: FramePacket) -> tuple[Detection, ...]:
        self.calls.append((packet.camera_id, packet.frame_id))
        self.called.set()
        if packet.camera_id == self.block_camera and not self.entered_block.is_set():
            self.entered_block.set()
            if not self.release_block.wait(timeout=2.0):
                raise TimeoutError("test did not release blocking detector")
        if self.fail_once_for == (packet.camera_id, packet.frame_id):
            self.fail_once_for = None
            raise RuntimeError("intentional inference failure")
        return (person_detection(packet),)


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()

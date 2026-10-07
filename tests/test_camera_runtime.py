from __future__ import annotations

import threading
import time
from pathlib import Path

from industrial_safety.video import (
    CameraConfig,
    CameraManager,
    CameraState,
    CameraWorker,
    SourceFrame,
    VideoSource,
)


class FakeSource(VideoSource):
    def __init__(self, *, fail_open: bool = False) -> None:
        self.fail_open = fail_open
        self.opened = False
        self.release_called = False
        self.read_count = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self.opened

    @property
    def fps(self) -> float:
        return 100.0

    @property
    def width(self) -> int:
        return 2

    @property
    def height(self) -> int:
        return 2

    @property
    def frame_count(self) -> int | None:
        return None

    @property
    def position(self) -> int:
        return self.read_count

    @property
    def loop_count(self) -> int:
        return 0

    def open(self) -> None:
        if self.fail_open:
            raise RuntimeError("intentional test failure")
        self.opened = True

    def read(self) -> SourceFrame:
        if not self.opened:
            raise RuntimeError("not open")
        with self._lock:
            frame_number = self.read_count
            self.read_count += 1
        return SourceFrame(
            image=frame_number,
            source_timestamp_ms=frame_number * 10.0,
            source_frame_number=frame_number,
            loop_count=0,
        )

    def release(self) -> None:
        self.release_called = True
        self.opened = False


def config(camera_id: str) -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        name="Test Camera",
        zone_id="test_zone",
        source_type="fake",
        source_path=Path("unused"),
        enabled=True,
    )


def wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_camera_shutdown_releases_source() -> None:
    source = FakeSource()
    worker = CameraWorker(config("CAM-TEST"), source)
    worker.start()
    assert wait_until(lambda: worker.health().frames_read >= 2)

    worker.stop()
    worker.join(timeout=1.0)

    assert not worker.is_alive
    assert source.release_called
    assert worker.latest_frame.is_closed
    assert worker.health().state is CameraState.STOPPED


def test_one_camera_failure_does_not_stop_other_workers() -> None:
    sources = {
        "CAM-BAD": FakeSource(fail_open=True),
        "CAM-GOOD": FakeSource(),
    }
    manager = CameraManager(
        [config("CAM-BAD"), config("CAM-GOOD")],
        source_factory=lambda camera: sources[camera.camera_id],
    )
    try:
        manager.start_all()
        assert wait_until(lambda: manager.health()["CAM-BAD"].state is CameraState.ERROR)
        assert wait_until(lambda: manager.health()["CAM-GOOD"].frames_read >= 3)
        assert manager.workers["CAM-GOOD"].is_alive
        assert manager.health()["CAM-GOOD"].state is CameraState.RUNNING
    finally:
        manager.stop_all(join_timeout=1.0)

    assert sources["CAM-BAD"].release_called
    assert sources["CAM-GOOD"].release_called

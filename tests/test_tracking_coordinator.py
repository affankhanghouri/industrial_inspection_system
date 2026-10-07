from __future__ import annotations

import time

from industrial_safety.inference import LatestInferenceResult
from industrial_safety.tracking import (
    CameraTrackingProcessor,
    PersonTracker,
    TrackingCoordinator,
)

from tracking_helpers import inference_result


class FakeTracker(PersonTracker):
    def __init__(self, camera_id: str, *, fail_once: bool = False) -> None:
        self._camera_id = camera_id
        self.fail_once = fail_once
        self.reset_calls = 0
        self.close_called = False
        self.frames: list[int] = []

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def update(self, result):
        self.frames.append(result.frame_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("intentional tracking failure")
        return ()

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_called = True


def wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_tracking_error_isolated_from_other_camera() -> None:
    inputs = {"CAM-BAD": LatestInferenceResult(), "CAM-GOOD": LatestInferenceResult()}
    bad = FakeTracker("CAM-BAD", fail_once=True)
    good = FakeTracker("CAM-GOOD")
    coordinator = TrackingCoordinator(
        inputs,
        processors={
            "CAM-BAD": CameraTrackingProcessor("CAM-BAD", bad),
            "CAM-GOOD": CameraTrackingProcessor("CAM-GOOD", good),
        },
        idle_wait_seconds=0.001,
    )
    inputs["CAM-BAD"].publish(inference_result("CAM-BAD", 1, 0.0))
    inputs["CAM-GOOD"].publish(inference_result("CAM-GOOD", 1, 0.0))
    try:
        coordinator.start()
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-BAD"].tracking_errors == 1
        )
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-GOOD"].tracking_updates == 1
        )
        assert coordinator.is_alive
    finally:
        coordinator.stop()
        coordinator.join(timeout=1.0)

    assert coordinator.fatal_error is None
    assert good.frames == [1]


def test_tracking_consumes_only_newest_result_and_stops_cleanly() -> None:
    input_buffer = LatestInferenceResult()
    tracker = FakeTracker("CAM-1")
    coordinator = TrackingCoordinator(
        {"CAM-1": input_buffer},
        processors={
            "CAM-1": CameraTrackingProcessor("CAM-1", tracker),
        },
        idle_wait_seconds=0.001,
    )
    input_buffer.publish(inference_result("CAM-1", 10, 0.0))
    input_buffer.publish(inference_result("CAM-1", 20, 0.5))
    coordinator.start()
    assert wait_until(lambda: coordinator.camera_stats()["CAM-1"].tracking_updates == 1)

    coordinator.stop()
    coordinator.join(timeout=1.0)

    assert tracker.frames == [20]
    assert tracker.reset_calls >= 1
    assert tracker.close_called
    assert not coordinator.is_alive
    assert coordinator.results["CAM-1"].is_closed

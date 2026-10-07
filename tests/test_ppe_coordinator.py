from __future__ import annotations

import threading
import time

from industrial_safety.ppe import (
    HelmetTrackAssociator,
    PPECoordinator,
    PPEDetector,
    load_ppe_config,
)
from industrial_safety.tracking import LatestTrackingResult

from ppe_helpers import helmet, person_track, tracking_result, video_frame


class RecordingPPEDetector(PPEDetector):
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def detect(self, packet):
        self.calls.append((packet.camera_id, packet.frame_id))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("intentional PPE failure")
        return (
            helmet(
                (30, 5, 70, 45),
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
            ),
        )

    def close(self) -> None:
        self.closed = True


def wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def make_coordinator(
    inputs,
    frames,
    detector,
    *,
    enabled=("CAM-001",),
    minimum_interval_seconds=0.0,
):
    return PPECoordinator(
        inputs,
        lambda camera_id: frames.get(camera_id),
        detector,
        HelmetTrackAssociator(load_ppe_config().association),
        enabled_camera_ids=enabled,
        minimum_interval_seconds=minimum_interval_seconds,
        idle_wait_seconds=0.001,
    )


def test_ppe_disabled_camera_does_not_run_inference() -> None:
    inputs = {
        "CAM-001": LatestTrackingResult(),
        "CAM-002": LatestTrackingResult(),
    }
    frames = {
        "CAM-001": video_frame("CAM-001", 10),
        "CAM-002": video_frame("CAM-002", 10),
    }
    detector = RecordingPPEDetector()
    coordinator = make_coordinator(inputs, frames, detector)
    inputs["CAM-002"].publish(
        tracking_result(
            (person_track(1, (0, 0, 100, 200), camera_id="CAM-002"),),
            camera_id="CAM-002",
        )
    )
    try:
        coordinator.start()
        time.sleep(0.03)
        assert detector.calls == []
        inputs["CAM-001"].publish(
            tracking_result(
                (
                    person_track(
                        1, (0, 0, 100, 200), camera_id="CAM-001"
                    ),
                ),
                camera_id="CAM-001",
            )
        )
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-001"].successful_inference_calls
            == 1
        )
    finally:
        coordinator.stop()
        coordinator.join(timeout=1.0)

    assert detector.calls == [("CAM-001", 10)]
    assert set(coordinator.inference_results) == {"CAM-001"}


def test_ppe_failure_is_isolated_and_new_frame_recovers() -> None:
    inputs = {
        camera_id: LatestTrackingResult()
        for camera_id in ("CAM-001", "CAM-002", "CAM-003")
    }
    frames = {"CAM-001": video_frame("CAM-001", 10)}
    detector = RecordingPPEDetector(fail_once=True)
    coordinator = make_coordinator(inputs, frames, detector)
    inputs["CAM-001"].publish(
        tracking_result(
            (person_track(1, (0, 0, 100, 200), camera_id="CAM-001"),),
            camera_id="CAM-001",
        )
    )
    try:
        coordinator.start()
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-001"].inference_errors == 1
        )
        assert coordinator.is_alive
        assert not inputs["CAM-002"].is_closed
        assert not inputs["CAM-003"].is_closed

        frames["CAM-001"] = video_frame("CAM-001", 20)
        inputs["CAM-001"].publish(
            tracking_result(
                (
                    person_track(
                        1,
                        (0, 0, 100, 200),
                        camera_id="CAM-001",
                        frame_id=20,
                    ),
                ),
                camera_id="CAM-001",
                frame_id=20,
            )
        )
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-001"].successful_inference_calls
            == 1
        )
    finally:
        coordinator.stop()
        coordinator.join(timeout=1.0)

    assert coordinator.fatal_error is None
    assert detector.closed
    assert not coordinator.is_alive
    assert coordinator.inference_results["CAM-001"].is_closed


def test_ppe_uses_only_exact_matching_frame() -> None:
    inputs = {"CAM-001": LatestTrackingResult()}
    frames = {"CAM-001": video_frame("CAM-001", 11)}
    detector = RecordingPPEDetector()
    coordinator = make_coordinator(inputs, frames, detector)
    inputs["CAM-001"].publish(
        tracking_result((person_track(1, (0, 0, 100, 200)),), frame_id=10)
    )
    try:
        coordinator.start()
        assert wait_until(
            lambda: coordinator.camera_stats()["CAM-001"].frame_mismatches == 1
        )
    finally:
        coordinator.stop()
        coordinator.join(timeout=1.0)

    assert detector.calls == []


def test_rate_cap_processes_newest_result_without_queueing_stale_frames() -> None:
    inputs = {"CAM-001": LatestTrackingResult()}
    frames = {"CAM-001": video_frame("CAM-001", 10)}
    detector = RecordingPPEDetector()
    coordinator = make_coordinator(
        inputs,
        frames,
        detector,
        minimum_interval_seconds=0.05,
    )
    inputs["CAM-001"].publish(
        tracking_result(
            (person_track(1, (0, 0, 100, 200), camera_id="CAM-001"),),
            camera_id="CAM-001",
        )
    )
    try:
        coordinator.start()
        assert wait_until(lambda: len(detector.calls) == 1)
        frames["CAM-001"] = video_frame("CAM-001", 20)
        inputs["CAM-001"].publish(
            tracking_result(
                (
                    person_track(
                        1,
                        (0, 0, 100, 200),
                        camera_id="CAM-001",
                        frame_id=20,
                    ),
                ),
                camera_id="CAM-001",
                frame_id=20,
            )
        )
        frames["CAM-001"] = video_frame("CAM-001", 30)
        inputs["CAM-001"].publish(
            tracking_result(
                (
                    person_track(
                        1,
                        (0, 0, 100, 200),
                        camera_id="CAM-001",
                        frame_id=30,
                    ),
                ),
                camera_id="CAM-001",
                frame_id=30,
            )
        )
        assert wait_until(lambda: len(detector.calls) == 2)
    finally:
        coordinator.stop()
        coordinator.join(timeout=1.0)

    assert detector.calls == [("CAM-001", 10), ("CAM-001", 30)]

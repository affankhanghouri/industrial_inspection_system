from industrial_safety.inference import RoundRobinInferenceScheduler

from inference_helpers import FakeProvider, RecordingDetector, frame, wait_until


def test_scheduler_does_not_repeat_consumed_frame() -> None:
    provider = FakeProvider()
    provider.latest_frame.publish(frame("CAM-1", 7))
    detector = RecordingDetector()
    scheduler = RoundRobinInferenceScheduler(
        {"CAM-1": provider}, detector, idle_wait_seconds=0.001
    )
    try:
        scheduler.start()
        assert wait_until(lambda: len(detector.calls) == 1)
        assert not wait_until(lambda: len(detector.calls) > 1, timeout=0.05)
    finally:
        scheduler.stop()
        scheduler.join(timeout=1.0)

    assert detector.calls == [("CAM-1", 7)]


def test_scheduler_rechecks_and_uses_newest_waiting_frame() -> None:
    first = FakeProvider()
    waiting = FakeProvider()
    first.latest_frame.publish(frame("CAM-1", 1))
    waiting.latest_frame.publish(frame("CAM-2", 150))
    detector = RecordingDetector()
    detector.block_camera = "CAM-1"
    scheduler = RoundRobinInferenceScheduler(
        {"CAM-1": first, "CAM-2": waiting},
        detector,
        idle_wait_seconds=0.001,
    )
    try:
        scheduler.start()
        assert detector.entered_block.wait(timeout=1.0)
        waiting.latest_frame.publish(frame("CAM-2", 158))
        detector.release_block.set()
        assert wait_until(lambda: any(call[0] == "CAM-2" for call in detector.calls))
    finally:
        scheduler.stop()
        scheduler.join(timeout=1.0)

    cam2_calls = [call for call in detector.calls if call[0] == "CAM-2"]
    assert cam2_calls[0] == ("CAM-2", 158)


def test_round_robin_eventually_services_every_camera() -> None:
    providers = {camera_id: FakeProvider() for camera_id in ("CAM-1", "CAM-2", "CAM-3")}
    for camera_id, provider in providers.items():
        provider.latest_frame.publish(frame(camera_id, 3))
    detector = RecordingDetector()
    scheduler = RoundRobinInferenceScheduler(
        providers, detector, idle_wait_seconds=0.001
    )
    try:
        scheduler.start()
        assert wait_until(lambda: len(detector.calls) == 3)
    finally:
        scheduler.stop()
        scheduler.join(timeout=1.0)

    assert detector.calls == [("CAM-1", 3), ("CAM-2", 3), ("CAM-3", 3)]
    assert all(stats.inference_calls == 1 for stats in scheduler.camera_stats().values())


def test_inference_error_is_recorded_and_newer_frame_recovers() -> None:
    provider = FakeProvider()
    provider.latest_frame.publish(frame("CAM-1", 10))
    detector = RecordingDetector()
    detector.fail_once_for = ("CAM-1", 10)
    scheduler = RoundRobinInferenceScheduler(
        {"CAM-1": provider}, detector, idle_wait_seconds=0.001
    )
    try:
        scheduler.start()
        assert wait_until(lambda: scheduler.camera_stats()["CAM-1"].inference_errors == 1)
        provider.latest_frame.publish(frame("CAM-1", 15))
        assert wait_until(
            lambda: scheduler.camera_stats()["CAM-1"].last_processed_frame_id == 15
        )
        assert scheduler.is_alive
    finally:
        scheduler.stop()
        scheduler.join(timeout=1.0)

    stats = scheduler.camera_stats()["CAM-1"]
    assert stats.inference_calls == 2
    assert stats.inference_errors == 1
    assert stats.last_processed_frame_id == 15


def test_scheduler_shutdown_closes_result_buffers() -> None:
    provider = FakeProvider()
    provider.latest_frame.publish(frame("CAM-1", 1))
    scheduler = RoundRobinInferenceScheduler(
        {"CAM-1": provider}, RecordingDetector(), idle_wait_seconds=0.001
    )
    scheduler.start()
    assert wait_until(lambda: scheduler.camera_stats()["CAM-1"].inference_calls == 1)

    scheduler.stop()
    scheduler.join(timeout=1.0)

    assert not scheduler.is_alive
    assert scheduler.results["CAM-1"].is_closed
    assert scheduler.fatal_error is None

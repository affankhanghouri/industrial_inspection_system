from industrial_safety.tracking import (
    CameraTrackingProcessor,
    TimeAwareIoUPersonTracker,
    load_tracking_config,
)

from tracking_helpers import inference_result


def processor(camera_id: str) -> CameraTrackingProcessor:
    tracker = TimeAwareIoUPersonTracker(camera_id, load_tracking_config().tracker)
    return CameraTrackingProcessor(camera_id, tracker)


def test_tracker_resets_when_mp4_loop_changes() -> None:
    camera = processor("CAM-1")
    first = camera.process(
        inference_result("CAM-1", 100, 3.0, ((0, 0, 20, 40),), loop_count=0)
    )
    looped = camera.process(
        inference_result(
            "CAM-1",
            116,
            3.5,
            ((100, 0, 120, 40),),
            loop_count=1,
            source_timestamp_ms=0.0,
        )
    )

    assert not first.tracker_was_reset
    assert looped.tracker_was_reset
    assert "source loop changed" in looped.reset_reason
    assert looped.tracks[0].track_id == 1


def test_tracker_resets_on_frame_id_regression() -> None:
    camera = processor("CAM-1")
    camera.process(inference_result("CAM-1", 100, 3.0, ((0, 0, 20, 40),)))

    restarted = camera.process(
        inference_result(
            "CAM-1", 2, 4.0, ((0, 0, 20, 40),), source_timestamp_ms=100.0
        )
    )

    assert restarted.tracker_was_reset
    assert "frame ID regressed" in restarted.reset_reason


def test_tracker_resets_on_source_timestamp_regression() -> None:
    camera = processor("CAM-1")
    camera.process(
        inference_result(
            "CAM-1",
            100,
            3.0,
            ((0, 0, 20, 40),),
            source_timestamp_ms=3000.0,
        )
    )

    restarted = camera.process(
        inference_result(
            "CAM-1",
            116,
            3.5,
            ((100, 0, 120, 40),),
            source_timestamp_ms=100.0,
        )
    )

    assert restarted.tracker_was_reset
    assert "source timestamp regressed" in restarted.reset_reason
    assert restarted.tracks[0].track_id == 1


def test_reset_of_one_camera_does_not_reset_another() -> None:
    first = processor("CAM-1")
    second = processor("CAM-2")
    first.process(inference_result("CAM-1", 10, 0.0, ((0, 0, 20, 40),)))
    second.process(inference_result("CAM-2", 10, 0.0, ((0, 0, 20, 40),)))

    first_result = first.process(
        inference_result("CAM-1", 20, 0.5, ((5, 0, 25, 40),), loop_count=1)
    )
    second_result = second.process(
        inference_result("CAM-2", 20, 0.5, ((5, 0, 25, 40),), loop_count=0)
    )

    assert first_result.tracker_was_reset
    assert not second_result.tracker_was_reset
    assert second_result.tracks[0].hits == 2

from datetime import datetime, timezone

from industrial_safety.tracking import LatestTrackingResult, TrackingResult

from tracking_helpers import inference_result


def tracking_result(frame_id: int) -> TrackingResult:
    now = datetime.now(timezone.utc)
    inference = inference_result("CAM-1", frame_id, float(frame_id))
    return TrackingResult(
        camera_id="CAM-1",
        frame_id=frame_id,
        inference_result=inference,
        tracks=(),
        tracking_started_at=now,
        tracking_finished_at=now,
        tracking_started_monotonic=1.0,
        tracking_finished_monotonic=1.001,
        tracking_ms=1.0,
        tracker_was_reset=False,
        reset_reason=None,
    )


def test_latest_tracking_result_overwrites_older_result() -> None:
    buffer = LatestTrackingResult()
    buffer.publish(tracking_result(10))
    buffer.publish(tracking_result(20))

    assert buffer.stored_result_count == 1
    assert buffer.get_latest(timeout=0).frame_id == 20
    assert buffer.get_latest(after_frame_id=20, timeout=0.01) is None

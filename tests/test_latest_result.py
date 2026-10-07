from datetime import datetime, timezone

from industrial_safety.inference import InferenceResult, LatestInferenceResult


def result(frame_id: int) -> InferenceResult:
    now = datetime.now(timezone.utc)
    return InferenceResult(
        camera_id="CAM-TEST",
        frame_id=frame_id,
        captured_at=now,
        source_timestamp_ms=float(frame_id),
        source_frame_number=frame_id,
        source_loop_count=0,
        inference_started_at=now,
        inference_finished_at=now,
        inference_started_monotonic=1.0,
        inference_finished_monotonic=1.1,
        detections=(),
        inference_ms=100.0,
        frame_age_at_start_ms=10.0,
        end_to_end_age_ms=110.0,
    )


def test_latest_inference_result_overwrites_stale_result() -> None:
    buffer = LatestInferenceResult()
    buffer.publish(result(10))
    buffer.publish(result(20))

    assert buffer.stored_result_count == 1
    assert buffer.get_latest(timeout=0).frame_id == 20
    assert buffer.get_latest(after_frame_id=20, timeout=0.01) is None

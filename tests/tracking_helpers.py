from __future__ import annotations

from datetime import datetime, timedelta, timezone

from industrial_safety.inference import Detection, InferenceResult


BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def detection(
    camera_id: str,
    frame_id: int,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
    observed_seconds: float = 0.0,
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_id=frame_id,
        class_id=0,
        class_name="person",
        confidence=confidence,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        detected_at=BASE_TIME + timedelta(seconds=observed_seconds),
    )


def inference_result(
    camera_id: str,
    frame_id: int,
    observed_seconds: float,
    boxes: tuple[tuple[float, float, float, float], ...] = (),
    *,
    loop_count: int = 0,
    source_timestamp_ms: float | None = None,
) -> InferenceResult:
    captured_at = BASE_TIME + timedelta(seconds=observed_seconds)
    detections = tuple(
        detection(
            camera_id,
            frame_id,
            bbox,
            observed_seconds=observed_seconds,
        )
        for bbox in boxes
    )
    return InferenceResult(
        camera_id=camera_id,
        frame_id=frame_id,
        captured_at=captured_at,
        source_timestamp_ms=(
            observed_seconds * 1000.0
            if source_timestamp_ms is None
            else source_timestamp_ms
        ),
        source_frame_number=frame_id,
        source_loop_count=loop_count,
        inference_started_at=captured_at + timedelta(milliseconds=10),
        inference_finished_at=captured_at + timedelta(milliseconds=110),
        inference_started_monotonic=1000.0 + observed_seconds + 0.010,
        inference_finished_monotonic=1000.0 + observed_seconds + 0.110,
        detections=detections,
        inference_ms=100.0,
        frame_age_at_start_ms=10.0,
        end_to_end_age_ms=110.0,
    )

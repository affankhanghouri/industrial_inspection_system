from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from industrial_safety.inference import Detection
from industrial_safety.ppe import (
    PPEInferenceResult,
    PPEObservation,
    PPEType,
)
from industrial_safety.tracking import PersonTrack, TrackingResult, TrackState

from inference_helpers import frame
from tracking_helpers import inference_result


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, confidence, classes) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(confidence)
        self.cls = FakeTensor(classes)

    def __len__(self) -> int:
        return len(self.conf.values)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeModel:
    names = {4: "hardhat", 6: "no-hardhat", 9: "person"}

    def __init__(self, boxes) -> None:
        self.boxes = boxes
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [FakeResult(self.boxes)]


def person_track(
    track_id: int,
    bbox: tuple[float, float, float, float],
    *,
    camera_id: str = "CAM-1",
    frame_id: int = 10,
    state: TrackState = TrackState.CONFIRMED,
    fresh: bool = True,
) -> PersonTrack:
    source = (
        Detection(
            camera_id=camera_id,
            frame_id=frame_id,
            class_id=0,
            class_name="person",
            confidence=0.9,
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
            detected_at=NOW,
        )
        if fresh
        else None
    )
    return PersonTrack(
        camera_id=camera_id,
        track_id=track_id,
        frame_id=frame_id,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        confidence=0.9,
        first_seen_at=NOW,
        last_seen_at=NOW,
        age_seconds=1.0,
        hits=3,
        missed_updates=0 if fresh else 1,
        state=state,
        source_detection=source,
    )


def tracking_result(
    tracks: tuple[PersonTrack, ...],
    *,
    camera_id: str = "CAM-1",
    frame_id: int = 10,
) -> TrackingResult:
    base = inference_result(camera_id, frame_id, 1.0)
    return TrackingResult(
        camera_id=camera_id,
        frame_id=frame_id,
        inference_result=base,
        tracks=tracks,
        tracking_started_at=NOW,
        tracking_finished_at=NOW,
        tracking_started_monotonic=1.0,
        tracking_finished_monotonic=1.001,
        tracking_ms=1.0,
        tracker_was_reset=False,
        reset_reason=None,
    )


def helmet(
    bbox: tuple[float, float, float, float],
    *,
    camera_id: str = "CAM-1",
    frame_id: int = 10,
    confidence: float = 0.9,
) -> PPEObservation:
    return PPEObservation(
        camera_id=camera_id,
        frame_id=frame_id,
        ppe_type=PPEType.HELMET,
        class_id=4,
        class_name="hardhat",
        confidence=confidence,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        observed_at=NOW,
    )


def ppe_result(
    observations: tuple[PPEObservation, ...] = (),
    *,
    camera_id: str = "CAM-1",
    frame_id: int = 10,
) -> PPEInferenceResult:
    return PPEInferenceResult(
        camera_id=camera_id,
        frame_id=frame_id,
        captured_at=NOW,
        source_timestamp_ms=1000.0,
        source_frame_number=frame_id,
        source_loop_count=0,
        observations=observations,
        inference_started_at=NOW,
        inference_finished_at=NOW,
        inference_started_monotonic=1.0,
        inference_finished_monotonic=1.1,
        inference_ms=100.0,
        frame_age_at_start_ms=10.0,
        end_to_end_age_ms=110.0,
    )


def video_frame(camera_id: str = "CAM-1", frame_id: int = 10):
    return frame(camera_id, frame_id)

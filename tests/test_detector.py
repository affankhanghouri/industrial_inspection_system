from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from industrial_safety.inference import (
    Detection,
    Detector,
    DetectorConfig,
    UltralyticsPersonDetector,
)

from inference_helpers import RecordingDetector, frame


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
    names = {0: "person", 2: "car"}

    def __init__(self, boxes) -> None:
        self.boxes = boxes
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [FakeResult(self.boxes)]


def detector_config() -> DetectorConfig:
    return DetectorConfig(
        backend="ultralytics",
        model_path=Path("unused.pt"),
        device="cpu",
        confidence=0.4,
        image_size=640,
        classes=("person",),
        cpu_threads=1,
        opencv_threads=1,
    )


def test_detector_interface_returns_structured_detections() -> None:
    detector: Detector = RecordingDetector()
    detections = detector.detect(frame("CAM-TEST", 4))

    assert isinstance(detections, tuple)
    assert isinstance(detections[0], Detection)
    assert detections[0].camera_id == "CAM-TEST"
    assert detections[0].frame_id == 4


def test_ultralytics_detector_filters_non_person_classes() -> None:
    model = FakeModel(
        FakeBoxes(
            xyxy=[[1, 2, 10, 12], [3, 4, 14, 15]],
            confidence=[0.91, 0.88],
            classes=[0, 2],
        )
    )
    detector = UltralyticsPersonDetector(
        detector_config(), model=model, configure_threads=False
    )

    detections = detector.detect(frame("CAM-TEST", 5))

    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].class_id == 0
    assert model.predict_kwargs["classes"] == [0]


def test_empty_detector_result_is_valid() -> None:
    model = FakeModel(FakeBoxes(xyxy=[], confidence=[], classes=[]))
    detector = UltralyticsPersonDetector(
        detector_config(), model=model, configure_threads=False
    )

    assert detector.detect(frame("CAM-TEST", 6)) == ()


def test_invalid_model_path_fails_initialization() -> None:
    with pytest.raises(FileNotFoundError, match="detector model does not exist"):
        UltralyticsPersonDetector(detector_config(), configure_threads=False)


def test_corrupt_frame_is_rejected() -> None:
    model = FakeModel(FakeBoxes(xyxy=[], confidence=[], classes=[]))
    detector = UltralyticsPersonDetector(
        detector_config(), model=model, configure_threads=False
    )
    packet = frame("CAM-TEST", 7)
    object.__setattr__(packet, "image", np.asarray([]))

    with pytest.raises(ValueError, match="invalid shape"):
        detector.detect(packet)

from pathlib import Path
from dataclasses import replace

from industrial_safety.ppe import (
    PPEDetector,
    PPEObservation,
    PPEType,
    UltralyticsPPEDetector,
    load_ppe_config,
)

from ppe_helpers import FakeBoxes, FakeModel, helmet, video_frame


class FakePPEDetector(PPEDetector):
    def detect(self, packet):
        return (
            helmet(
                (1, 1, 5, 5),
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
            ),
        )


def detector_config():
    return replace(
        load_ppe_config().detector,
        backend="ultralytics",
        model_path=Path("models/yolov8n_ppe_construction.pt").resolve(),
    )


def test_ppe_detector_interface_returns_structured_observation() -> None:
    detector: PPEDetector = FakePPEDetector()
    observations = detector.detect(video_frame(frame_id=4))

    assert isinstance(observations, tuple)
    assert isinstance(observations[0], PPEObservation)
    assert observations[0].ppe_type is PPEType.HELMET
    assert observations[0].center_x == 3.0


def test_ultralytics_ppe_detector_filters_non_helmet_classes() -> None:
    model = FakeModel(
        FakeBoxes(
            xyxy=[[1, 2, 10, 12], [3, 4, 14, 15]],
            confidence=[0.91, 0.88],
            classes=[4, 6],
        )
    )
    detector = UltralyticsPPEDetector(
        detector_config(), model=model, configure_threads=False
    )

    observations = detector.detect(video_frame(frame_id=5))

    assert len(observations) == 1
    assert observations[0].class_name == "hardhat"
    assert model.predict_kwargs["classes"] == [4]


def test_empty_helmet_detection_is_valid() -> None:
    detector = UltralyticsPPEDetector(
        detector_config(),
        model=FakeModel(FakeBoxes([], [], [])),
        configure_threads=False,
    )

    assert detector.detect(video_frame(frame_id=6)) == ()


def test_multiple_helmet_observations_are_valid() -> None:
    detector = UltralyticsPPEDetector(
        detector_config(),
        model=FakeModel(
            FakeBoxes(
                [[1, 1, 5, 5], [10, 1, 14, 5]],
                [0.9, 0.8],
                [4, 4],
            )
        ),
        configure_threads=False,
    )

    observations = detector.detect(video_frame())

    assert len(observations) == 2
    assert all(item.ppe_type is PPEType.HELMET for item in observations)


def test_model_path_is_a_local_checkpoint() -> None:
    path = detector_config().model_path
    assert path == Path("models/yolov8n_ppe_construction.pt").resolve()

"""Person detector implementations backed by ONNX Runtime or OpenVINO."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from industrial_safety.video.types import FramePacket

from .config import DetectorConfig
from .detector import Detector
from .export_runtime import ExportedYoloRuntime
from .types import Detection


class ExportedPersonDetector(Detector):
    def __init__(self, config: DetectorConfig) -> None:
        if config.backend not in {"onnxruntime", "openvino"}:
            raise ValueError(f"unsupported exported detector backend: {config.backend!r}")
        if config.device != "cpu":
            raise ValueError("exported detector requires device 'cpu'")
        self.config = config
        self._runtime = ExportedYoloRuntime(
            config.backend,
            config.model_path,
            image_size=config.image_size,
            cpu_threads=config.cpu_threads,
            confidence=config.confidence,
            requested_class_names=config.classes,
        )

    def warmup(self) -> None:
        self._runtime.warmup()

    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        if not isinstance(frame.image, np.ndarray):
            raise TypeError("frame image must be a NumPy array")
        if frame.image.ndim != 3 or frame.image.size == 0:
            raise ValueError(f"frame image has invalid shape: {frame.image.shape}")
        detected_at = datetime.now(timezone.utc)
        return tuple(
            Detection(
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                class_id=item.class_id,
                class_name=item.class_name,
                confidence=item.confidence,
                x1=item.x1,
                y1=item.y1,
                x2=item.x2,
                y2=item.y2,
                detected_at=detected_at,
            )
            for item in self._runtime.detect(frame.image)
        )


class ONNXPersonDetector(ExportedPersonDetector):
    pass


class OpenVINOPersonDetector(ExportedPersonDetector):
    pass


def create_person_detector(config: DetectorConfig) -> Detector:
    if config.backend == "ultralytics":
        from .ultralytics_detector import UltralyticsPersonDetector

        return UltralyticsPersonDetector(config)
    if config.backend == "onnxruntime":
        return ONNXPersonDetector(config)
    if config.backend == "openvino":
        return OpenVINOPersonDetector(config)
    raise ValueError(f"unsupported person detector backend: {config.backend!r}")

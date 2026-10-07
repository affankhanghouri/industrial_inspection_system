"""Helmet detector implementations backed by ONNX Runtime or OpenVINO."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from industrial_safety.inference.export_runtime import ExportedYoloRuntime
from industrial_safety.video.types import FramePacket

from .config import PPEDetectorConfig
from .detector import PPEDetector
from .types import PPEObservation, PPEType


class ExportedPPEDetector(PPEDetector):
    def __init__(self, config: PPEDetectorConfig) -> None:
        if config.backend not in {"onnxruntime", "openvino"}:
            raise ValueError(f"unsupported exported PPE backend: {config.backend!r}")
        if config.device != "cpu":
            raise ValueError("exported PPE detector requires device 'cpu'")
        if config.enabled_types != ("helmet",):
            raise ValueError("exported PPE detector currently exposes only helmets")
        self.config = config
        self._runtime = ExportedYoloRuntime(
            config.backend,
            config.model_path,
            image_size=config.image_size,
            cpu_threads=config.cpu_threads,
            confidence=config.confidence,
            requested_class_names=config.helmet_class_names,
        )

    @property
    def helmet_class_ids(self) -> frozenset[int]:
        return self._runtime.class_ids

    def warmup(self) -> None:
        self._runtime.warmup()

    def detect(self, frame: FramePacket) -> tuple[PPEObservation, ...]:
        if not isinstance(frame.image, np.ndarray):
            raise TypeError("PPE frame image must be a NumPy array")
        if frame.image.ndim != 3 or frame.image.size == 0:
            raise ValueError(f"PPE frame image has invalid shape: {frame.image.shape}")
        observed_at = datetime.now(timezone.utc)
        return tuple(
            PPEObservation(
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                ppe_type=PPEType.HELMET,
                class_id=item.class_id,
                class_name=item.class_name,
                confidence=item.confidence,
                x1=item.x1,
                y1=item.y1,
                x2=item.x2,
                y2=item.y2,
                observed_at=observed_at,
            )
            for item in self._runtime.detect(frame.image)
        )


class ONNXPPEDetector(ExportedPPEDetector):
    pass


class OpenVINOPPEDetector(ExportedPPEDetector):
    pass


def create_ppe_detector(config: PPEDetectorConfig) -> PPEDetector:
    if config.backend == "ultralytics":
        from .ultralytics_detector import UltralyticsPPEDetector

        return UltralyticsPPEDetector(config)
    if config.backend == "onnxruntime":
        return ONNXPPEDetector(config)
    if config.backend == "openvino":
        return OpenVINOPPEDetector(config)
    raise ValueError(f"unsupported PPE detector backend: {config.backend!r}")

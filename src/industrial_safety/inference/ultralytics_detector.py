"""Ultralytics implementation of person-only detection."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from industrial_safety.video.types import FramePacket

from .config import DetectorConfig
from .detector import Detector
from .types import Detection

ModelFactory = Callable[[str], Any]


class UltralyticsPersonDetector(Detector):
    """One Ultralytics model instance restricted to the COCO person class."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        model: Any | None = None,
        model_factory: ModelFactory | None = None,
        configure_threads: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if config.backend != "ultralytics":
            raise ValueError(f"unsupported detector backend: {config.backend!r}")
        if config.device != "cpu":
            raise ValueError("this CPU development runtime requires detector device 'cpu'")
        self.config = config
        self._logger = logger or logging.getLogger(__name__)

        if configure_threads:
            self._configure_cpu_threads()

        if model is None:
            if not config.model_path.is_file():
                raise FileNotFoundError(f"detector model does not exist: {config.model_path}")
            if model_factory is None:
                from ultralytics import YOLO

                model_factory = lambda path: YOLO(path, task="detect")
            try:
                model = model_factory(str(config.model_path))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to initialize Ultralytics model {config.model_path}: {exc}"
                ) from exc
        self._model = model

        names = getattr(self._model, "names", None)
        if not isinstance(names, (dict, list, tuple)):
            raise RuntimeError("Ultralytics model does not expose class names")
        items = names.items() if isinstance(names, dict) else enumerate(names)
        requested = set(config.classes)
        self._class_names = {int(class_id): str(name) for class_id, name in items}
        self._class_ids = {
            class_id
            for class_id, name in self._class_names.items()
            if name.strip().lower() in requested
        }
        if not self._class_ids:
            raise RuntimeError(
                f"model contains none of the requested classes: {sorted(requested)}"
            )

    def warmup(self) -> None:
        image = np.zeros(
            (self.config.image_size, self.config.image_size, 3), dtype=np.uint8
        )
        self._model.predict(
            source=image,
            device=self.config.device,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            classes=sorted(self._class_ids),
            verbose=False,
            save=False,
        )

    def _configure_cpu_threads(self) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(self.config.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.config.cpu_threads))
        import torch

        torch.set_num_threads(self.config.cpu_threads)
        try:
            torch.set_num_interop_threads(self.config.cpu_threads)
        except RuntimeError:
            self._logger.debug("PyTorch interop thread count was already initialized")
        cv2.setNumThreads(self.config.opencv_threads)

    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        if not isinstance(frame.image, np.ndarray):
            raise TypeError("frame image must be a NumPy array")
        if frame.image.ndim != 3 or frame.image.size == 0:
            raise ValueError(f"frame image has invalid shape: {frame.image.shape}")

        results = self._model.predict(
            source=frame.image,
            device=self.config.device,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            classes=sorted(self._class_ids),
            verbose=False,
            save=False,
        )
        if not results:
            return ()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()

        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        detected_at = datetime.now(timezone.utc)
        detections: list[Detection] = []
        for xyxy, confidence, raw_class_id in zip(
            coordinates, confidences, class_ids, strict=True
        ):
            class_id = int(raw_class_id)
            # Retain a defensive filter even though Ultralytics receives classes=[person].
            if class_id not in self._class_ids:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            detections.append(
                Detection(
                    camera_id=frame.camera_id,
                    frame_id=frame.frame_id,
                    class_id=class_id,
                    class_name=self._class_names[class_id],
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    detected_at=detected_at,
                )
            )
        return tuple(detections)

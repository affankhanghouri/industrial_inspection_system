"""Ultralytics PPE detector restricted to configured helmet classes."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from industrial_safety.video.types import FramePacket

from .config import PPEDetectorConfig
from .detector import PPEDetector
from .types import PPEObservation, PPEType

ModelFactory = Callable[[str], Any]


class UltralyticsPPEDetector(PPEDetector):
    def __init__(
        self,
        config: PPEDetectorConfig,
        *,
        model: Any | None = None,
        model_factory: ModelFactory | None = None,
        configure_threads: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if config.backend != "ultralytics":
            raise ValueError(f"unsupported PPE backend: {config.backend!r}")
        if config.device != "cpu":
            raise ValueError("this development runtime requires PPE device 'cpu'")
        if config.enabled_types != ("helmet",):
            raise ValueError("UltralyticsPPEDetector currently exposes only helmets")
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        if configure_threads:
            self._configure_cpu_threads()

        if model is None:
            if not config.model_path.is_file():
                raise FileNotFoundError(f"PPE model does not exist: {config.model_path}")
            if model_factory is None:
                from ultralytics import YOLO

                model_factory = lambda path: YOLO(path, task="detect")
            try:
                model = model_factory(str(config.model_path))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to initialize PPE model {config.model_path}: {exc}"
                ) from exc
        self._model = model

        names = getattr(model, "names", None)
        if not isinstance(names, (dict, list, tuple)):
            raise RuntimeError("PPE model does not expose class names")
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self._class_names = {int(class_id): str(name) for class_id, name in items}
        requested = set(config.helmet_class_names)
        self._helmet_class_ids = {
            class_id
            for class_id, name in self._class_names.items()
            if name.strip().lower() in requested
        }
        if not self._helmet_class_ids:
            raise RuntimeError(
                "PPE model contains none of the configured helmet classes: "
                f"{sorted(requested)}; available={sorted(self._class_names.values())}"
            )

    @property
    def helmet_class_ids(self) -> frozenset[int]:
        return frozenset(self._helmet_class_ids)

    def warmup(self) -> None:
        image = np.zeros(
            (self.config.image_size, self.config.image_size, 3), dtype=np.uint8
        )
        self._model.predict(
            source=image,
            device=self.config.device,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            classes=sorted(self._helmet_class_ids),
            verbose=False,
            save=False,
        )

    def detect(self, frame: FramePacket) -> tuple[PPEObservation, ...]:
        if not isinstance(frame.image, np.ndarray):
            raise TypeError("PPE frame image must be a NumPy array")
        if frame.image.ndim != 3 or frame.image.size == 0:
            raise ValueError(f"PPE frame image has invalid shape: {frame.image.shape}")
        results = self._model.predict(
            source=frame.image,
            device=self.config.device,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            classes=sorted(self._helmet_class_ids),
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
        observed_at = datetime.now(timezone.utc)
        observations: list[PPEObservation] = []
        for xyxy, confidence, raw_class_id in zip(
            coordinates, confidences, class_ids, strict=True
        ):
            class_id = int(raw_class_id)
            if class_id not in self._helmet_class_ids:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            observations.append(
                PPEObservation(
                    camera_id=frame.camera_id,
                    frame_id=frame.frame_id,
                    ppe_type=PPEType.HELMET,
                    class_id=class_id,
                    class_name=self._class_names[class_id],
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    observed_at=observed_at,
                )
            )
        return tuple(observations)

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

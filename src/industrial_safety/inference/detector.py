"""Detector backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from industrial_safety.video.types import FramePacket

from .types import Detection


class Detector(ABC):
    """Convert one current camera frame into structured detections."""

    @abstractmethod
    def detect(self, frame: FramePacket) -> tuple[Detection, ...]:
        """Detect relevant objects in ``frame`` or raise a descriptive exception."""

    def warmup(self) -> None:
        """Initialize lazy backend state before live frames arrive, if needed."""

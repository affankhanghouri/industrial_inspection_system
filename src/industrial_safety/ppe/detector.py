"""Backend-neutral PPE detector contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from industrial_safety.video.types import FramePacket

from .types import PPEObservation


class PPEDetector(ABC):
    @abstractmethod
    def detect(self, frame: FramePacket) -> tuple[PPEObservation, ...]:
        """Return structured PPE observations without retaining the image."""

    def warmup(self) -> None:
        """Run an optional backend warm-up."""

    def close(self) -> None:
        """Release backend resources, if any."""

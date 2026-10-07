"""Backend-neutral person tracker contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from industrial_safety.inference.types import InferenceResult

from .types import PersonTrack


class PersonTracker(ABC):
    """Maintain anonymous local trajectories for exactly one camera."""

    @property
    @abstractmethod
    def camera_id(self) -> str: ...

    @abstractmethod
    def update(self, result: InferenceResult) -> tuple[PersonTrack, ...]:
        """Apply one observation and return all non-expired track states."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all state. Track IDs may restart after reset."""

    def close(self) -> None:
        """Release resources held by the tracker backend, if any."""

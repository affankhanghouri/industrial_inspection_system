"""Video source contract independent of OpenCV capture details."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import SourceFrame


class VideoSource(ABC):
    """Interface implemented by MP4 now and RTSP sources later."""

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def frame_count(self) -> int | None: ...

    @property
    @abstractmethod
    def position(self) -> int: ...

    @property
    @abstractmethod
    def loop_count(self) -> int: ...

    @abstractmethod
    def open(self) -> None:
        """Open the underlying source or raise a descriptive exception."""

    @abstractmethod
    def read(self) -> SourceFrame:
        """Return the next decoded frame or raise a descriptive exception."""

    @abstractmethod
    def release(self) -> None:
        """Release the source. This method must be safe to call repeatedly."""

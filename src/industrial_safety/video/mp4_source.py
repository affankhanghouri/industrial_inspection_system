"""OpenCV-backed, continuously looping MP4 source."""

from __future__ import annotations

from pathlib import Path

import cv2

from .source import VideoSource
from .types import SourceFrame


class MP4VideoSource(VideoSource):
    """Decode an MP4 and rewind it transparently when EOF is reached."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._capture: cv2.VideoCapture | None = None
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count: int | None = None
        self._position = 0
        self._loop_count = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int | None:
        return self._frame_count

    @property
    def position(self) -> int:
        return self._position

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def open(self) -> None:
        self.release()
        if not self.path.is_file():
            raise FileNotFoundError(f"video source does not exist: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video source: {self.path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(
                f"video has invalid metadata: fps={fps}, width={width}, height={height}"
            )

        self._capture = capture
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_count = raw_frame_count if raw_frame_count > 0 else None
        self._position = 0
        self._loop_count = 0

    def read(self) -> SourceFrame:
        if not self.is_open or self._capture is None:
            raise RuntimeError("video source is not open")

        ok, image = self._capture.read()
        if not ok or image is None:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError(f"failed to rewind video source: {self.path}")
            self._loop_count += 1
            self._position = 0
            ok, image = self._capture.read()
            if not ok or image is None:
                raise RuntimeError(f"failed to decode video source after rewind: {self.path}")

        source_frame_number = max(0, int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        source_timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        self._position = source_frame_number + 1
        return SourceFrame(
            image=image,
            source_timestamp_ms=max(0.0, source_timestamp_ms),
            source_frame_number=source_frame_number,
            loop_count=self._loop_count,
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

"""Single-slot tracking result storage."""

from __future__ import annotations

import threading

from .types import TrackingResult


class LatestTrackingResult:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: TrackingResult | None = None
        self._closed = False

    def publish(self, result: TrackingResult) -> None:
        with self._condition:
            if self._closed:
                return
            self._latest = result
            self._condition.notify_all()

    def get_latest(
        self,
        *,
        after_frame_id: int | None = None,
        timeout: float | None = None,
    ) -> TrackingResult | None:
        def available() -> bool:
            return self._closed or (
                self._latest is not None
                and (after_frame_id is None or self._latest.frame_id > after_frame_id)
            )

        with self._condition:
            if not available():
                self._condition.wait_for(available, timeout=timeout)
            if self._latest is None:
                return None
            if after_frame_id is not None and self._latest.frame_id <= after_frame_id:
                return None
            return self._latest

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def stored_result_count(self) -> int:
        with self._condition:
            return int(self._latest is not None)

"""Lifecycle management for independent camera workers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from .camera_worker import CameraWorker
from .config import CameraConfig
from .mp4_source import MP4VideoSource
from .source import VideoSource
from .types import CameraHealth

SourceFactory = Callable[[CameraConfig], VideoSource]


def _default_source_factory(config: CameraConfig) -> VideoSource:
    if config.source_type == "mp4":
        return MP4VideoSource(config.source_path)
    raise ValueError(
        f"unsupported source_type {config.source_type!r} for {config.camera_id}"
    )


class CameraManager:
    def __init__(
        self,
        configs: Iterable[CameraConfig],
        *,
        source_factory: SourceFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        factory = source_factory or _default_source_factory
        enabled = [config for config in configs if config.enabled]
        self._workers: dict[str, CameraWorker] = {}
        for config in enabled:
            if config.camera_id in self._workers:
                raise ValueError(f"duplicate camera_id: {config.camera_id}")
            self._workers[config.camera_id] = CameraWorker(
                config,
                factory(config),
                logger=self._logger,
            )

    @property
    def workers(self) -> dict[str, CameraWorker]:
        return dict(self._workers)

    def start_all(self) -> None:
        self._logger.info("starting %d camera worker(s)", len(self._workers))
        for worker in self._workers.values():
            worker.start()

    def stop_all(self, *, join_timeout: float = 5.0) -> None:
        for worker in self._workers.values():
            worker.stop()
        for worker in self._workers.values():
            worker.join(timeout=join_timeout)
            if worker.is_alive:
                self._logger.warning(
                    "[%s] did not stop within %.1f seconds",
                    worker.config.camera_id,
                    join_timeout,
                )

    def health(self) -> dict[str, CameraHealth]:
        return {
            camera_id: worker.health()
            for camera_id, worker in self._workers.items()
        }

    def __enter__(self) -> CameraManager:
        self.start_all()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_all()

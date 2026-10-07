"""Latest-only coordinator from tracks/PPE through rules and persisted events."""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable

from industrial_safety.events.service import EventService
from industrial_safety.events.types import SafetyEvent
from industrial_safety.ppe.latest_result import LatestPPEAssociationResult
from industrial_safety.rules import RuleEngine
from industrial_safety.tracking.latest_result import LatestTrackingResult
from industrial_safety.video.types import FramePacket
from industrial_safety.zones import ZoneEngine, ZoneEvaluation

FrameProvider = Callable[[str], FramePacket | None]


class SafetyCoordinator:
    def __init__(
        self,
        tracking_results: dict[str, LatestTrackingResult],
        ppe_results: dict[str, LatestPPEAssociationResult],
        zone_engine: ZoneEngine,
        rule_engine: RuleEngine,
        event_service: EventService,
        tracking_frame_provider: FrameProvider,
        ppe_frame_provider: FrameProvider,
        *,
        idle_wait_seconds: float = 0.005,
        logger: logging.Logger | None = None,
    ) -> None:
        self._tracking_results = dict(tracking_results)
        self._ppe_results = dict(ppe_results)
        self._zone_engine = zone_engine
        self._rule_engine = rule_engine
        self._event_service = event_service
        self._tracking_frame_provider = tracking_frame_provider
        self._ppe_frame_provider = ppe_frame_provider
        self._idle_wait_seconds = idle_wait_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._last_tracking: dict[str, int] = {}
        self._last_ppe: dict[str, int] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._events_created = 0
        self._errors = 0
        self._recent_processing_ms: deque[float] = deque(maxlen=1000)
        self._latest_zone_evaluations: dict[str, ZoneEvaluation] = {}
        self._latest_events: dict[str, SafetyEvent] = {}
        self._fatal_error: str | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fatal_error(self) -> str | None:
        with self._lock:
            return self._fatal_error

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            values = tuple(self._recent_processing_ms)
            return {
                "events_created": self._events_created,
                "errors": self._errors,
                "mean_processing_ms": statistics.mean(values) if values else 0.0,
                "p95_processing_ms": (
                    sorted(values)[max(0, int(len(values) * 0.95) - 1)]
                    if values
                    else 0.0
                ),
            }

    def latest_zone_evaluation(self, camera_id: str) -> ZoneEvaluation | None:
        with self._lock:
            return self._latest_zone_evaluations.get(camera_id)

    def latest_event(self, camera_id: str) -> SafetyEvent | None:
        with self._lock:
            return self._latest_events.get(camera_id)

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        self._last_tracking.clear()
        self._last_ppe.clear()
        with self._lock:
            self._latest_zone_evaluations.clear()
            self._latest_events.clear()
        self._thread = threading.Thread(
            target=self._run, name="safety-event-coordinator", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                processed = False
                for camera_id, buffer in self._tracking_results.items():
                    tracking = buffer.get_latest(timeout=0.0)
                    if tracking is None or self._last_tracking.get(camera_id) == tracking.frame_id:
                        continue
                    self._last_tracking[camera_id] = tracking.frame_id
                    processed = True
                    self._process_tracking(camera_id, tracking)
                for camera_id, buffer in self._ppe_results.items():
                    result = buffer.get_latest(timeout=0.0)
                    if result is None or self._last_ppe.get(camera_id) == result.frame_id:
                        continue
                    self._last_ppe[camera_id] = result.frame_id
                    processed = True
                    self._process_ppe(camera_id, result)
                if not processed:
                    self._stop_event.wait(self._idle_wait_seconds)
        except Exception as exc:
            with self._lock:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("safety coordinator failed")
        finally:
            self._logger.info("safety coordinator stopped")

    def _process_tracking(self, camera_id: str, tracking: object) -> None:
        started = time.perf_counter()
        try:
            evaluation = self._zone_engine.evaluate(tracking)  # type: ignore[arg-type]
            with self._lock:
                self._latest_zone_evaluations[camera_id] = evaluation
            if evaluation.reset_applied:
                self._rule_engine.reset_camera(camera_id)
            for transition in evaluation.transitions:
                for match in self._rule_engine.evaluate_zone(transition):
                    frame = self._frame_for(
                        camera_id, match.track.frame_id, self._tracking_frame_provider
                    )
                    if frame is None:
                        raise RuntimeError("no frame available for zone-event evidence")
                    event = self._event_service.create(match, frame)
                    if event is not None:
                        with self._lock:
                            self._events_created += 1
                            self._latest_events[camera_id] = event
        except Exception:
            with self._lock:
                self._errors += 1
            self._logger.exception("[%s] zone/rule processing failed", camera_id)
        finally:
            with self._lock:
                self._recent_processing_ms.append(
                    (time.perf_counter() - started) * 1000.0
                )

    def _process_ppe(self, camera_id: str, result: object) -> None:
        started = time.perf_counter()
        try:
            for match in self._rule_engine.evaluate_ppe(result):  # type: ignore[arg-type]
                frame = self._frame_for(
                    camera_id, match.track.frame_id, self._ppe_frame_provider
                )
                if frame is None:
                    raise RuntimeError("no frame available for PPE-event evidence")
                event = self._event_service.create(match, frame)
                if event is not None:
                    with self._lock:
                        self._events_created += 1
                        self._latest_events[camera_id] = event
        except Exception:
            with self._lock:
                self._errors += 1
            self._logger.exception("[%s] PPE temporal/rule processing failed", camera_id)
        finally:
            with self._lock:
                self._recent_processing_ms.append(
                    (time.perf_counter() - started) * 1000.0
                )

    @staticmethod
    def _frame_for(
        camera_id: str, frame_id: int, provider: FrameProvider
    ) -> FramePacket | None:
        frame = provider(camera_id)
        if frame is None:
            return None
        # Evidence is bounded to the newest retained frame. Exact frames are
        # preferred; a newer live frame is acceptable if the single slot advanced.
        return frame if frame.frame_id >= frame_id else None

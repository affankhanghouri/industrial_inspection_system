"""Camera-local bottom-center polygon state and debounced transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass

from industrial_safety.tracking.types import TrackState, TrackingResult

from .types import (
    Zone,
    ZoneEvaluation,
    ZoneObservation,
    ZoneState,
    ZoneTransition,
    ZoneTransitionType,
)


def point_in_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    """Return true for points inside or on an edge of a simple polygon."""

    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-6 and min(x1, x2) <= x <= max(x1, x2) and min(
            y1, y2
        ) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            intersect_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersect_x:
                inside = not inside
    return inside


@dataclass(slots=True)
class _TrackZoneState:
    stable_inside: bool
    candidate_inside: bool
    candidate_count: int = 0


class ZoneEngine:
    def __init__(self, zones: tuple[Zone, ...]) -> None:
        self._zones_by_camera: dict[str, tuple[Zone, ...]] = {}
        for camera_id in {zone.camera_id for zone in zones if zone.enabled}:
            self._zones_by_camera[camera_id] = tuple(
                zone for zone in zones if zone.enabled and zone.camera_id == camera_id
            )
        self._states: dict[tuple[str, str, int], _TrackZoneState] = {}

    def reset_camera(self, camera_id: str) -> None:
        self._states = {
            key: value for key, value in self._states.items() if key[0] != camera_id
        }

    def evaluate(self, tracking: TrackingResult) -> ZoneEvaluation:
        started = time.perf_counter()
        reset = tracking.tracker_was_reset
        if reset:
            self.reset_camera(tracking.camera_id)
        observations: list[ZoneObservation] = []
        transitions: list[ZoneTransition] = []
        present_track_ids = {track.track_id for track in tracking.tracks}
        for track in tracking.tracks:
            if track.state is not TrackState.CONFIRMED or track.source_detection is None:
                continue
            for zone in self._zones_by_camera.get(tracking.camera_id, ()):
                inside = point_in_polygon(track.bottom_center, zone.polygon)
                key = (tracking.camera_id, zone.zone_id, track.track_id)
                state = self._states.get(key)
                if state is None:
                    state = _TrackZoneState(inside, inside)
                    self._states[key] = state
                elif inside == state.stable_inside:
                    state.candidate_inside = inside
                    state.candidate_count = 0
                else:
                    if state.candidate_inside != inside:
                        state.candidate_inside = inside
                        state.candidate_count = 1
                    else:
                        state.candidate_count += 1
                    required = (
                        zone.enter_confirmations if inside else zone.exit_confirmations
                    )
                    if state.candidate_count >= required:
                        state.stable_inside = inside
                        state.candidate_count = 0
                        transitions.append(
                            ZoneTransition(
                                camera_id=tracking.camera_id,
                                frame_id=tracking.frame_id,
                                zone=zone,
                                track=track,
                                transition=(
                                    ZoneTransitionType.ENTER
                                    if inside
                                    else ZoneTransitionType.EXIT
                                ),
                                occurred_at=tracking.inference_result.captured_at,
                            )
                        )
                observations.append(
                    ZoneObservation(
                        camera_id=tracking.camera_id,
                        frame_id=tracking.frame_id,
                        zone=zone,
                        track=track,
                        state=(
                            ZoneState.INSIDE
                            if state.stable_inside
                            else ZoneState.OUTSIDE
                        ),
                        observed_at=tracking.inference_result.captured_at,
                    )
                )
        active_keys = {
            (tracking.camera_id, zone.zone_id, track_id)
            for zone in self._zones_by_camera.get(tracking.camera_id, ())
            for track_id in present_track_ids
        }
        self._states = {
            key: value
            for key, value in self._states.items()
            if key[0] != tracking.camera_id or key in active_keys
        }
        return ZoneEvaluation(
            camera_id=tracking.camera_id,
            frame_id=tracking.frame_id,
            observations=tuple(observations),
            transitions=tuple(transitions),
            evaluation_ms=(time.perf_counter() - started) * 1000.0,
            reset_applied=reset,
        )

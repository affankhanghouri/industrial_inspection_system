"""Lightweight, elapsed-time-aware tracking by detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from industrial_safety.inference.types import Detection, InferenceResult

from .config import TrackerConfig
from .tracker import PersonTracker
from .types import PersonTrack, TrackState

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class _TrackRecord:
    track_id: int
    bbox: BBox
    velocity: BBox
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    first_seen_monotonic: float
    last_seen_monotonic: float
    hits: int
    missed_updates: int
    state: TrackState
    source_detection: Detection | None


class TimeAwareIoUPersonTracker(PersonTracker):
    """Associate detections using predicted overlap and normalized center distance.

    This is intentionally small and camera-local. It uses no appearance/ReID model.
    """

    def __init__(self, camera_id: str, config: TrackerConfig) -> None:
        if not camera_id:
            raise ValueError("camera_id is required")
        if config.backend != "time_aware_iou":
            raise ValueError(f"unsupported tracker backend: {config.backend!r}")
        self._camera_id = camera_id
        self.config = config
        self._tracks: dict[int, _TrackRecord] = {}
        self._next_track_id = 1
        self._last_observation_monotonic: float | None = None
        self._closed = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def update(self, result: InferenceResult) -> tuple[PersonTrack, ...]:
        if self._closed:
            raise RuntimeError("tracker is closed")
        if result.camera_id != self._camera_id:
            raise ValueError(
                f"tracker {self._camera_id} cannot process {result.camera_id}"
            )
        observed = result.observed_monotonic
        if (
            self._last_observation_monotonic is not None
            and observed < self._last_observation_monotonic
        ):
            raise ValueError("observation monotonic time moved backwards")
        self._last_observation_monotonic = observed

        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if observed - track.last_seen_monotonic > self.config.maximum_lost_seconds
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]

        detections = [
            detection
            for detection in result.detections
            if detection.class_name.lower() == "person"
            and detection.confidence >= self.config.minimum_detection_confidence
        ]
        predicted = {
            track_id: self._predict_bbox(track, observed)
            for track_id, track in self._tracks.items()
        }
        matches = self._associate(predicted, detections)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_detection_indices = {index for _, index in matches}

        for track_id, detection_index in matches:
            self._apply_detection(
                self._tracks[track_id], detections[detection_index], observed, result
            )

        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            track.missed_updates += 1
            track.state = TrackState.LOST
            track.source_detection = None

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detection_indices:
                self._create_track(detection, observed, result)

        output: list[PersonTrack] = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            bbox = (
                track.bbox
                if track.state is not TrackState.LOST
                else self._predict_bbox(track, observed)
            )
            output.append(self._as_person_track(track, bbox, result, observed))
        return tuple(output)

    def reset(self) -> None:
        if self._closed:
            return
        self._tracks.clear()
        self._next_track_id = 1
        self._last_observation_monotonic = None

    def close(self) -> None:
        self.reset()
        self._closed = True

    def _associate(
        self,
        predicted: dict[int, BBox],
        detections: list[Detection],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, predicted_bbox in predicted.items():
            for detection_index, detection in enumerate(detections):
                detection_bbox = _detection_bbox(detection)
                overlap = _iou(predicted_bbox, detection_bbox)
                distance = _normalized_center_distance(predicted_bbox, detection_bbox)
                if (
                    overlap < self.config.minimum_iou
                    and distance > self.config.maximum_center_distance
                ):
                    continue
                distance_score = max(
                    0.0, 1.0 - distance / self.config.maximum_center_distance
                )
                score = 2.0 * overlap + distance_score
                candidates.append((score, track_id, detection_index))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            matches.append((track_id, detection_index))
        return matches

    def _apply_detection(
        self,
        track: _TrackRecord,
        detection: Detection,
        observed: float,
        result: InferenceResult,
    ) -> None:
        new_bbox = _detection_bbox(detection)
        elapsed = max(1e-6, observed - track.last_seen_monotonic)
        instantaneous_velocity = tuple(
            (new - old) / elapsed for old, new in zip(track.bbox, new_bbox, strict=True)
        )
        smoothing = self.config.velocity_smoothing
        track.velocity = tuple(
            smoothing * old + (1.0 - smoothing) * new
            for old, new in zip(
                track.velocity, instantaneous_velocity, strict=True
            )
        )
        track.bbox = new_bbox
        track.confidence = detection.confidence
        track.last_seen_at = result.captured_at
        track.last_seen_monotonic = observed
        track.hits += 1
        track.missed_updates = 0
        track.state = (
            TrackState.CONFIRMED
            if track.hits >= self.config.minimum_confirmed_hits
            else TrackState.TENTATIVE
        )
        track.source_detection = detection

    def _create_track(
        self,
        detection: Detection,
        observed: float,
        result: InferenceResult,
    ) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _TrackRecord(
            track_id=track_id,
            bbox=_detection_bbox(detection),
            velocity=(0.0, 0.0, 0.0, 0.0),
            confidence=detection.confidence,
            first_seen_at=result.captured_at,
            last_seen_at=result.captured_at,
            first_seen_monotonic=observed,
            last_seen_monotonic=observed,
            hits=1,
            missed_updates=0,
            state=(
                TrackState.CONFIRMED
                if self.config.minimum_confirmed_hits <= 1
                else TrackState.TENTATIVE
            ),
            source_detection=detection,
        )

    def _predict_bbox(self, track: _TrackRecord, observed: float) -> BBox:
        elapsed = min(
            max(0.0, observed - track.last_seen_monotonic),
            self.config.maximum_prediction_seconds,
        )
        predicted = tuple(
            coordinate + velocity * elapsed
            for coordinate, velocity in zip(track.bbox, track.velocity, strict=True)
        )
        x1, y1, x2, y2 = predicted
        if x2 <= x1 or y2 <= y1:
            return track.bbox
        return predicted

    def _as_person_track(
        self,
        track: _TrackRecord,
        bbox: BBox,
        result: InferenceResult,
        observed: float,
    ) -> PersonTrack:
        return PersonTrack(
            camera_id=self._camera_id,
            track_id=track.track_id,
            frame_id=result.frame_id,
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
            confidence=track.confidence,
            first_seen_at=track.first_seen_at,
            last_seen_at=track.last_seen_at,
            age_seconds=max(0.0, observed - track.first_seen_monotonic),
            hits=track.hits,
            missed_updates=track.missed_updates,
            state=track.state,
            source_detection=track.source_detection,
        )


def _detection_bbox(detection: Detection) -> BBox:
    return (detection.x1, detection.y1, detection.x2, detection.y2)


def _iou(first: BBox, second: BBox) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(first: BBox, second: BBox) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = math.hypot(
        first_center[0] - second_center[0], first_center[1] - second_center[1]
    )
    first_diagonal = math.hypot(first[2] - first[0], first[3] - first[1])
    second_diagonal = math.hypot(second[2] - second[0], second[3] - second[1])
    return distance / max(1.0, first_diagonal, second_diagonal)

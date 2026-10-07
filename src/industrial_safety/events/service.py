"""Event deduplication, evidence capture, and persistence."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from industrial_safety.rules.types import RuleMatch
from industrial_safety.storage.sqlite_store import SQLiteEventStore
from industrial_safety.video.types import FramePacket

from .evidence import EvidenceRecorder
from .types import EventStatus, SafetyEvent


class EventService:
    def __init__(
        self,
        store: SQLiteEventStore,
        evidence_root: str | Path,
        *,
        person_model_version: str,
        ppe_model_version: str,
        evidence_recorder: EvidenceRecorder | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.store = store
        self.evidence_root = Path(evidence_root).resolve()
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.person_model_version = person_model_version
        self.ppe_model_version = ppe_model_version
        self.evidence_recorder = evidence_recorder
        self._logger = logger or logging.getLogger(__name__)

    def create(self, match: RuleMatch, frame: FramePacket) -> SafetyEvent | None:
        latest = self.store.latest_for_rule(match.camera_id, match.rule.rule_id)
        if latest is not None:
            elapsed = (match.detected_at - latest.detected_at).total_seconds()
            if elapsed < match.rule.cooldown_seconds:
                self._logger.info(
                    "[%s] suppressed duplicate %s inside %.0fs cooldown",
                    match.camera_id,
                    match.rule.event_type,
                    match.rule.cooldown_seconds,
                )
                return None
        event_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        event_directory = Path(match.detected_at.date().isoformat()) / event_id
        relative_path = event_directory / "snapshot.jpg"
        relative_clip_path = event_directory / "clip.mp4"
        destination = self.evidence_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        annotated = self._annotate(frame.image, match)
        if not cv2.imwrite(str(destination), annotated):
            raise RuntimeError(f"failed to save evidence snapshot: {destination}")
        clip_path: str | None = None
        if self.evidence_recorder is not None and self.evidence_recorder.trigger(
            event_id,
            match.camera_id,
            relative_clip_path,
            triggered_monotonic=frame.captured_monotonic,
        ):
            clip_path = str(relative_clip_path)
        event = SafetyEvent(
            event_id=event_id,
            event_type=match.rule.event_type,
            status=EventStatus.OPEN,
            severity=match.rule.severity,
            detected_at=match.detected_at,
            camera_id=match.camera_id,
            zone_id=match.zone_id,
            track_id=match.track.track_id,
            rule_id=match.rule.rule_id,
            rule_version=match.rule.version,
            person_model_version=self.person_model_version,
            ppe_model_version=(
                self.ppe_model_version if match.ppe_result is not None else None
            ),
            confidence=match.confidence,
            supporting_observations=match.supporting_observations,
            snapshot_path=str(relative_path),
            clip_path=clip_path,
            created_at=created_at,
        )
        self.store.add(event)
        self._logger.warning(
            "[%s] confirmed %s track=%d event=%s evidence=%s",
            event.camera_id,
            event.event_type,
            event.track_id,
            event.event_id,
            destination,
        )
        return event

    def _annotate(self, image: np.ndarray, match: RuleMatch) -> np.ndarray:
        canvas = image.copy()
        transition = match.zone_transition
        if transition is not None:
            points = np.asarray(transition.zone.polygon, dtype=np.int32)
            cv2.polylines(canvas, [points], True, (0, 165, 255), 3)
        track = match.track
        cv2.rectangle(
            canvas,
            (round(track.x1), round(track.y1)),
            (round(track.x2), round(track.y2)),
            (0, 0, 220),
            3,
        )
        label = f"Track {track.track_id}"
        cv2.rectangle(
            canvas,
            (round(track.x1), max(48, round(track.y1) - 30)),
            (round(track.x1) + 105, max(75, round(track.y1))),
            (0, 0, 170),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (round(track.x1) + 6, max(69, round(track.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        event_label = match.rule.event_type.replace("_", " ").title()
        banner = f"{event_label}  |  {match.camera_id}  |  {match.detected_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (24, 29, 36), -1)
        cv2.putText(
            canvas,
            banner,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

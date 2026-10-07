"""Persistent safety-event records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    event_id: str
    event_type: str
    status: EventStatus
    severity: str
    detected_at: datetime
    camera_id: str
    zone_id: str | None
    track_id: int
    rule_id: str
    rule_version: str
    person_model_version: str
    ppe_model_version: str | None
    confidence: float | None
    supporting_observations: int
    snapshot_path: str
    clip_path: str | None
    created_at: datetime
    reviewed_at: datetime | None = None
    review_action: str | None = None
    review_reason: str | None = None

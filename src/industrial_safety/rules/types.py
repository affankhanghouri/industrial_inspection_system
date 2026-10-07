"""Rule configuration and matched-rule records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from industrial_safety.ppe.types import PPEAssociationResult
from industrial_safety.tracking.types import PersonTrack
from industrial_safety.zones.types import ZoneTransition


class RuleType(str, Enum):
    HELMET_REQUIRED = "helmet_required"
    LOADING_AREA_ENTRY = "loading_area_entry"
    RESTRICTED_AREA_ENTRY = "restricted_area_entry"


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    version: str
    camera_id: str
    rule_type: RuleType
    event_type: str
    severity: str
    enabled: bool
    cooldown_seconds: float
    zone_id: str | None = None
    minimum_unobserved: int | None = None
    observation_window_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RuleMatch:
    rule: Rule
    camera_id: str
    track: PersonTrack
    detected_at: datetime
    zone_id: str | None
    confidence: float | None
    supporting_observations: int
    ppe_result: PPEAssociationResult | None = None
    zone_transition: ZoneTransition | None = None

"""Immutable polygon-zone observation records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from industrial_safety.tracking.types import PersonTrack


class ZoneState(str, Enum):
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"


class ZoneTransitionType(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: tuple[tuple[float, float], ...]
    enabled: bool
    enter_confirmations: int
    exit_confirmations: int


@dataclass(frozen=True, slots=True)
class ZoneObservation:
    camera_id: str
    frame_id: int
    zone: Zone
    track: PersonTrack
    state: ZoneState
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ZoneTransition:
    camera_id: str
    frame_id: int
    zone: Zone
    track: PersonTrack
    transition: ZoneTransitionType
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ZoneEvaluation:
    camera_id: str
    frame_id: int
    observations: tuple[ZoneObservation, ...]
    transitions: tuple[ZoneTransition, ...]
    evaluation_ms: float
    reset_applied: bool

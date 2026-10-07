"""Tracking configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    backend: str
    minimum_detection_confidence: float
    minimum_iou: float
    maximum_center_distance: float
    maximum_lost_seconds: float
    minimum_confirmed_hits: int
    maximum_prediction_seconds: float
    velocity_smoothing: float


@dataclass(frozen=True, slots=True)
class TrackingRuntimeConfig:
    idle_wait_seconds: float


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    tracker: TrackerConfig
    runtime: TrackingRuntimeConfig


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_tracking_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> TrackingConfig:
    root = Path(project_root).resolve() if project_root else _default_project_root()
    path = Path(config_path).resolve() if config_path else root / "config" / "tracking.yaml"
    with path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}
    if not isinstance(document, dict):
        raise ValueError("tracking configuration must be a mapping")
    tracker = document.get("tracker")
    runtime = document.get("runtime")
    if not isinstance(tracker, dict) or not isinstance(runtime, dict):
        raise ValueError("tracking configuration requires tracker and runtime mappings")

    backend = tracker.get("backend")
    if not isinstance(backend, str) or backend.strip().lower() != "time_aware_iou":
        raise ValueError("Milestone 3 supports backend 'time_aware_iou'")

    def number(key: str, *, minimum: float, maximum: float | None = None) -> float:
        value = tracker.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"tracker {key} must be numeric")
        converted = float(value)
        if converted < minimum or (maximum is not None and converted > maximum):
            raise ValueError(f"tracker {key} is outside the allowed range")
        return converted

    minimum_confirmed_hits = tracker.get("minimum_confirmed_hits")
    if not isinstance(minimum_confirmed_hits, int) or minimum_confirmed_hits <= 0:
        raise ValueError("tracker minimum_confirmed_hits must be a positive integer")
    idle_wait_ms = runtime.get("idle_wait_ms")
    if not isinstance(idle_wait_ms, (int, float)) or float(idle_wait_ms) <= 0:
        raise ValueError("tracking runtime idle_wait_ms must be greater than zero")

    return TrackingConfig(
        tracker=TrackerConfig(
            backend=backend.strip().lower(),
            minimum_detection_confidence=number(
                "minimum_detection_confidence", minimum=0.0, maximum=1.0
            ),
            minimum_iou=number("minimum_iou", minimum=0.0, maximum=1.0),
            maximum_center_distance=number("maximum_center_distance", minimum=0.01),
            maximum_lost_seconds=number("maximum_lost_seconds", minimum=0.01),
            minimum_confirmed_hits=minimum_confirmed_hits,
            maximum_prediction_seconds=number(
                "maximum_prediction_seconds", minimum=0.0
            ),
            velocity_smoothing=number(
                "velocity_smoothing", minimum=0.0, maximum=1.0
            ),
        ),
        runtime=TrackingRuntimeConfig(
            idle_wait_seconds=float(idle_wait_ms) / 1000.0
        ),
    )

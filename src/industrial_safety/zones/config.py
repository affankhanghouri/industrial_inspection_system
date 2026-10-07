"""Zone configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import Zone


def load_zones(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[Zone, ...]:
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[3]
    )
    path = Path(config_path).resolve() if config_path else root / "config/zones.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = document.get("zones") if isinstance(document, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("zone configuration requires a non-empty zones list")
    zones: list[Zone] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("each zone must be a mapping")
        zone_id = _text(raw, "zone_id")
        if zone_id in seen:
            raise ValueError(f"duplicate zone_id: {zone_id}")
        seen.add(zone_id)
        polygon_raw = raw.get("polygon")
        if not isinstance(polygon_raw, list) or len(polygon_raw) < 3:
            raise ValueError(f"zone {zone_id} polygon requires at least three points")
        polygon: list[tuple[float, float]] = []
        for point in polygon_raw:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)
            ):
                raise ValueError(f"zone {zone_id} has an invalid polygon point")
            polygon.append((float(point[0]), float(point[1])))
        zones.append(
            Zone(
                zone_id=zone_id,
                camera_id=_text(raw, "camera_id"),
                name=_text(raw, "name"),
                zone_type=_text(raw, "zone_type"),
                polygon=tuple(polygon),
                enabled=raw.get("enabled") is True,
                enter_confirmations=_positive_int(raw, "enter_confirmations"),
                exit_confirmations=_positive_int(raw, "exit_confirmations"),
            )
        )
    return tuple(zones)


def _text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"zone {key} must be non-empty text")
    return value.strip()


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"zone {key} must be a positive integer")
    return value

"""Camera configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    name: str
    zone_id: str
    source_type: str
    source_path: Path
    enabled: bool = True


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _required_text(entry: dict[str, Any], field: str, index: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"camera entry {index} has invalid {field!r}")
    return value.strip()


def load_camera_configs(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> list[CameraConfig]:
    """Load cameras and resolve relative sources against the project root."""

    root = Path(project_root).resolve() if project_root else _default_project_root()
    path = Path(config_path).resolve() if config_path else root / "config" / "cameras.yaml"
    with path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}

    entries = document.get("cameras")
    if not isinstance(entries, list):
        raise ValueError("configuration must contain a 'cameras' list")

    cameras: list[CameraConfig] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"camera entry {index} must be a mapping")
        camera_id = _required_text(entry, "camera_id", index)
        if camera_id in seen_ids:
            raise ValueError(f"duplicate camera_id: {camera_id}")
        seen_ids.add(camera_id)

        source_type = _required_text(entry, "source_type", index).lower()
        source_value = Path(_required_text(entry, "source_path", index)).expanduser()
        source_path = source_value if source_value.is_absolute() else root / source_value
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"camera entry {index} has non-boolean 'enabled'")

        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                name=_required_text(entry, "name", index),
                zone_id=_required_text(entry, "zone_id", index),
                source_type=source_type,
                source_path=source_path.resolve(),
                enabled=enabled,
            )
        )
    return cameras

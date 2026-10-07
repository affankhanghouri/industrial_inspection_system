"""PPE detector, association, and scheduling configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PPEDetectorConfig:
    backend: str
    model_path: Path
    device: str
    confidence: float
    image_size: int
    enabled_types: tuple[str, ...]
    helmet_class_names: tuple[str, ...]
    cpu_threads: int
    opencv_threads: int


@dataclass(frozen=True, slots=True)
class PPEAssociationConfig:
    head_region_fraction: float
    horizontal_expansion: float
    vertical_expansion: float
    minimum_helmet_overlap: float
    maximum_center_distance: float
    minimum_score: float
    ambiguity_margin: float


@dataclass(frozen=True, slots=True)
class PPERuntimeConfig:
    enabled_camera_ids: tuple[str, ...]
    minimum_interval_seconds: float
    idle_wait_seconds: float


@dataclass(frozen=True, slots=True)
class PPEConfig:
    detector: PPEDetectorConfig
    association: PPEAssociationConfig
    runtime: PPERuntimeConfig


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_ppe_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> PPEConfig:
    root = Path(project_root).resolve() if project_root else _default_project_root()
    path = Path(config_path).resolve() if config_path else root / "config" / "ppe.yaml"
    with path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}
    if not isinstance(document, dict):
        raise ValueError("PPE configuration must be a mapping")
    detector = document.get("detector")
    association = document.get("association")
    runtime = document.get("runtime")
    if not all(isinstance(section, dict) for section in (detector, association, runtime)):
        raise ValueError("PPE configuration requires detector, association, and runtime")

    def text(mapping: dict[str, Any], key: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"PPE {key} must be non-empty text")
        return value.strip()

    def number(
        mapping: dict[str, Any],
        key: str,
        *,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        value = mapping.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"PPE {key} must be numeric")
        converted = float(value)
        if converted < minimum or (maximum is not None and converted > maximum):
            raise ValueError(f"PPE {key} is outside the allowed range")
        return converted

    def positive_int(mapping: dict[str, Any], key: str) -> int:
        value = mapping.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"PPE {key} must be a positive integer")
        return value

    def text_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
        value = mapping.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"PPE {key} must be a non-empty list")
        normalized = tuple(
            item.strip().lower()
            for item in value
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(value):
            raise ValueError(f"PPE {key} must contain non-empty text values")
        return normalized

    backend = text(detector, "backend").lower()
    if backend not in {"ultralytics", "onnxruntime", "openvino"}:
        raise ValueError(f"unsupported PPE backend: {backend!r}")
    enabled_types = text_tuple(detector, "enabled_types")
    if enabled_types != ("helmet",):
        raise ValueError("Milestone 4 enables only the helmet PPE type")
    model_value = Path(text(detector, "model")).expanduser()
    model_path = model_value if model_value.is_absolute() else root / model_value
    if not model_path.exists():
        raise FileNotFoundError(f"PPE model does not exist: {model_path}")

    enabled_cameras = text_tuple(runtime, "enabled_cameras")
    if len(set(enabled_cameras)) != len(enabled_cameras):
        raise ValueError("PPE enabled_cameras contains duplicates")
    # Camera identifiers are case-sensitive even though model class names are not.
    raw_enabled = runtime["enabled_cameras"]
    enabled_camera_ids = tuple(item.strip() for item in raw_enabled)

    return PPEConfig(
        detector=PPEDetectorConfig(
            backend=backend,
            model_path=model_path.resolve(),
            device=text(detector, "device").lower(),
            confidence=number(detector, "confidence", minimum=0.0, maximum=1.0),
            image_size=positive_int(detector, "image_size"),
            enabled_types=enabled_types,
            helmet_class_names=text_tuple(detector, "helmet_class_names"),
            cpu_threads=positive_int(detector, "cpu_threads"),
            opencv_threads=positive_int(detector, "opencv_threads"),
        ),
        association=PPEAssociationConfig(
            head_region_fraction=number(
                association, "head_region_fraction", minimum=0.05, maximum=1.0
            ),
            horizontal_expansion=number(
                association, "horizontal_expansion", minimum=0.0, maximum=1.0
            ),
            vertical_expansion=number(
                association, "vertical_expansion", minimum=0.0, maximum=1.0
            ),
            minimum_helmet_overlap=number(
                association, "minimum_helmet_overlap", minimum=0.0, maximum=1.0
            ),
            maximum_center_distance=number(
                association, "maximum_center_distance", minimum=0.01
            ),
            minimum_score=number(
                association, "minimum_score", minimum=0.0, maximum=1.0
            ),
            ambiguity_margin=number(
                association, "ambiguity_margin", minimum=0.0, maximum=1.0
            ),
        ),
        runtime=PPERuntimeConfig(
            enabled_camera_ids=enabled_camera_ids,
            minimum_interval_seconds=number(
                runtime, "minimum_interval_seconds", minimum=0.0
            ),
            idle_wait_seconds=number(runtime, "idle_wait_ms", minimum=0.1)
            / 1000.0,
        ),
    )

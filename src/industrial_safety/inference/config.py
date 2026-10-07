"""Inference configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    backend: str
    model_path: Path
    device: str
    confidence: float
    image_size: int
    classes: tuple[str, ...]
    cpu_threads: int
    opencv_threads: int


@dataclass(frozen=True, slots=True)
class InferenceRuntimeConfig:
    scheduling: str
    idle_wait_seconds: float


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    detector: DetectorConfig
    runtime: InferenceRuntimeConfig


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"inference configuration must contain a {key!r} mapping")
    return value


def _text(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid inference setting: {key}")
    return value.strip()


def load_inference_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> InferenceConfig:
    root = Path(project_root).resolve() if project_root else _default_project_root()
    path = Path(config_path).resolve() if config_path else root / "config" / "inference.yaml"
    with path.open("r", encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}
    if not isinstance(document, dict):
        raise ValueError("inference configuration must be a mapping")

    detector = _mapping(document, "detector")
    runtime = _mapping(document, "runtime")
    model_value = Path(_text(detector, "model")).expanduser()
    model_path = model_value if model_value.is_absolute() else root / model_value
    confidence = detector.get("confidence")
    image_size = detector.get("image_size")
    cpu_threads = detector.get("cpu_threads")
    opencv_threads = detector.get("opencv_threads")
    class_values = detector.get("classes")
    idle_wait_ms = runtime.get("idle_wait_ms")

    backend = _text(detector, "backend").lower()
    if backend not in {"ultralytics", "onnxruntime", "openvino"}:
        raise ValueError(f"unsupported detector backend: {backend!r}")
    if not model_path.exists():
        raise FileNotFoundError(f"detector model does not exist: {model_path}")
    if not isinstance(confidence, (int, float)) or not 0.0 < float(confidence) <= 1.0:
        raise ValueError("detector confidence must be in (0, 1]")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("detector image_size must be a positive integer")
    if not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError("detector cpu_threads must be a positive integer")
    if not isinstance(opencv_threads, int) or opencv_threads <= 0:
        raise ValueError("detector opencv_threads must be a positive integer")
    if not isinstance(class_values, list) or not class_values:
        raise ValueError("detector classes must be a non-empty list")
    classes = tuple(str(value).strip().lower() for value in class_values)
    if any(not value for value in classes):
        raise ValueError("detector classes cannot contain empty values")
    if classes != ("person",):
        raise ValueError("Milestone 2 supports person detection only")
    if not isinstance(idle_wait_ms, (int, float)) or float(idle_wait_ms) <= 0:
        raise ValueError("runtime idle_wait_ms must be greater than zero")

    scheduling = _text(runtime, "scheduling").lower()
    if scheduling != "round_robin":
        raise ValueError("Milestone 2 supports round_robin scheduling only")

    return InferenceConfig(
        detector=DetectorConfig(
            backend=backend,
            model_path=model_path.resolve(),
            device=_text(detector, "device").lower(),
            confidence=float(confidence),
            image_size=image_size,
            classes=classes,
            cpu_threads=cpu_threads,
            opencv_threads=opencv_threads,
        ),
        runtime=InferenceRuntimeConfig(
            scheduling=scheduling,
            idle_wait_seconds=float(idle_wait_ms) / 1000.0,
        ),
    )

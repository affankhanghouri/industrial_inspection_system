"""Rule configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import Rule, RuleType


def load_rules(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> tuple[Rule, ...]:
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[3]
    )
    path = Path(config_path).resolve() if config_path else root / "config/rules.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = document.get("rules") if isinstance(document, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("rule configuration requires a non-empty rules list")
    rules: list[Rule] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("each rule must be a mapping")
        rule_id = _text(raw, "rule_id")
        if rule_id in seen:
            raise ValueError(f"duplicate rule_id: {rule_id}")
        seen.add(rule_id)
        rule_type = RuleType(_text(raw, "rule_type"))
        minimum = raw.get("minimum_unobserved")
        window = raw.get("observation_window_seconds")
        if rule_type is RuleType.HELMET_REQUIRED:
            if not isinstance(minimum, int) or minimum < 2:
                raise ValueError("helmet rule minimum_unobserved must be at least two")
            if not isinstance(window, (int, float)) or float(window) <= 0:
                raise ValueError("helmet rule observation window must be positive")
        rules.append(
            Rule(
                rule_id=rule_id,
                version=_text(raw, "version"),
                camera_id=_text(raw, "camera_id"),
                rule_type=rule_type,
                event_type=_text(raw, "event_type"),
                severity=_text(raw, "severity"),
                enabled=raw.get("enabled") is True,
                cooldown_seconds=_number(raw, "cooldown_seconds"),
                zone_id=(
                    _text(raw, "zone_id") if raw.get("zone_id") is not None else None
                ),
                minimum_unobserved=minimum,
                observation_window_seconds=(float(window) if window is not None else None),
            )
        )
    return tuple(rules)


def _text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rule {key} must be non-empty text")
    return value.strip()


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise ValueError(f"rule {key} must be non-negative")
    return float(value)

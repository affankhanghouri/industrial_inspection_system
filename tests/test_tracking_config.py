from pathlib import Path

from industrial_safety.tracking import load_tracking_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tracking_config_loads() -> None:
    config = load_tracking_config(project_root=PROJECT_ROOT)

    assert config.tracker.backend == "time_aware_iou"
    assert config.tracker.maximum_lost_seconds == 1.75
    assert config.tracker.minimum_confirmed_hits == 2
    assert config.runtime.idle_wait_seconds == 0.005

from pathlib import Path

from industrial_safety.ppe import load_ppe_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ppe_configuration_loads_model_and_camera_scope() -> None:
    config = load_ppe_config(project_root=PROJECT_ROOT)

    assert config.detector.backend == "openvino"
    assert config.detector.model_path.is_dir()
    assert config.detector.image_size == 512
    assert config.detector.cpu_threads == 2
    assert config.detector.enabled_types == ("helmet",)
    assert config.detector.helmet_class_names == ("hardhat",)
    assert config.runtime.enabled_camera_ids == ("CAM-001",)
    assert config.runtime.minimum_interval_seconds == 1.0

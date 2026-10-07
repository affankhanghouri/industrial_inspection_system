from pathlib import Path

from industrial_safety.video import load_camera_configs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_camera_configuration_loads_and_resolves_paths() -> None:
    cameras = load_camera_configs(project_root=PROJECT_ROOT)

    assert [camera.camera_id for camera in cameras] == ["CAM-001", "CAM-002", "CAM-003"]
    assert all(camera.enabled for camera in cameras)
    assert all(camera.source_type == "mp4" for camera in cameras)
    assert all(camera.source_path.is_absolute() for camera in cameras)
    assert all(camera.source_path.is_file() for camera in cameras)

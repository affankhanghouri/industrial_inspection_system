from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_safety.video import CameraConfig
from industrial_safety.video import load_camera_configs
from scripts.run_system import (
    DisplayTelemetry,
    MULTI_DISPLAY_WINDOW,
    MultiCameraDisplayState,
    _display_requests_shutdown,
    compose_multi_camera_preview,
    display_mode,
    display_window_name,
    parse_args,
    select_camera_configs,
)


def camera(camera_id: str, *, enabled: bool = True) -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        name=camera_id,
        zone_id="test-zone",
        source_type="mp4",
        source_path=Path("test.mp4"),
        enabled=enabled,
    )


def test_camera_and_display_arguments_parse() -> None:
    args = parse_args(
        [
            "--display",
            "--camera",
            "CAM-002",
            "--display-screenshot",
            "debug/cam-002.png",
        ]
    )

    assert args.display is True
    assert args.camera == "CAM-002"
    assert args.display_screenshot == Path("debug/cam-002.png")


def test_launcher_defaults_are_unchanged() -> None:
    args = parse_args([])

    assert args.display is False
    assert args.camera is None
    assert args.no_api is False


def test_display_without_camera_selects_multi_camera_mode() -> None:
    assert display_mode(parse_args(["--display"])) == "multi"
    assert display_mode(parse_args(["--display", "--camera", "CAM-001"])) == "single"
    assert display_mode(parse_args([])) == "headless"


def test_valid_camera_selection_contains_only_requested_camera() -> None:
    selected = select_camera_configs(
        (camera("CAM-001"), camera("CAM-002"), camera("CAM-003")),
        "CAM-002",
    )

    assert tuple(item.camera_id for item in selected) == ("CAM-002",)


def test_real_camera_ids_resolve_to_distinct_expected_sources() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configs = load_camera_configs(project_root=project_root)
    expected = {
        "CAM-001": "camera_01_helmet_violation.mp4",
        "CAM-002": "camera_02_loading_area.mp4",
        "CAM-003": "camera_03_restricted_area.mp4",
    }

    selected_paths = {
        camera_id: select_camera_configs(configs, camera_id)[0].source_path
        for camera_id in expected
    }

    assert len(set(selected_paths.values())) == 3
    assert {
        camera_id: path.name for camera_id, path in selected_paths.items()
    } == expected
    assert all(path.is_absolute() and path.is_file() for path in selected_paths.values())


def test_display_window_is_camera_specific() -> None:
    assert display_window_name("CAM-001") != display_window_name("CAM-002")
    assert display_window_name("CAM-002").endswith("[CAM-002]")
    assert MULTI_DISPLAY_WINDOW == "Industrial Safety - Multi Camera"


def test_tiled_compositor_accepts_three_frames_in_correct_positions() -> None:
    frames = {
        "CAM-001": np.full((90, 160, 3), (10, 20, 30), dtype=np.uint8),
        "CAM-002": np.full((90, 160, 3), (40, 50, 60), dtype=np.uint8),
        "CAM-003": np.full((90, 160, 3), (70, 80, 90), dtype=np.uint8),
    }
    labels = {camera_id: camera_id for camera_id in frames}

    output = compose_multi_camera_preview(frames, labels)

    assert output.shape == (720, 1280, 3)
    assert tuple(output[190, 320]) == (10, 20, 30)
    assert tuple(output[190, 960]) == (40, 50, 60)
    assert tuple(output[550, 640]) == (70, 80, 90)


def test_tiled_compositor_uses_matching_camera_labels(monkeypatch) -> None:
    texts = []
    original = cv2.putText

    def record_text(image, text, *args, **kwargs):
        texts.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    frames = {camera_id: np.zeros((90, 160, 3), dtype=np.uint8) for camera_id in ("CAM-001", "CAM-002", "CAM-003")}
    compose_multi_camera_preview(
        frames,
        {
            "CAM-001": "Refinery Area",
            "CAM-002": "Loading Area",
            "CAM-003": "Restricted Area",
        },
    )

    assert "CAM-001 | Refinery Area" in texts
    assert "CAM-002 | Loading Area" in texts
    assert "CAM-003 | Restricted Area" in texts


def test_tiled_compositor_handles_missing_camera_frame() -> None:
    output = compose_multi_camera_preview(
        {"CAM-001": None, "CAM-002": np.zeros((90, 160, 3), dtype=np.uint8), "CAM-003": None},
        {"CAM-001": "Refinery Area", "CAM-002": "Loading Area", "CAM-003": "Restricted Area"},
    )

    assert output.shape == (720, 1280, 3)
    assert np.any(output[:360, :640] != 0)


def test_multi_display_refreshes_independent_camera_tiles_round_robin() -> None:
    state = MultiCameraDisplayState()
    camera_ids = ("CAM-001", "CAM-002", "CAM-003")

    assert state.cameras_to_refresh(camera_ids) == camera_ids
    state.canvases.update({camera_id: object() for camera_id in camera_ids})
    assert state.cameras_to_refresh(camera_ids) == ("CAM-001", "CAM-002")
    assert state.cameras_to_refresh(camera_ids) == ("CAM-003", "CAM-001")


def test_invalid_camera_id_has_helpful_error() -> None:
    with pytest.raises(SystemExit, match="unknown camera ID 'CAM-999'.*CAM-001"):
        select_camera_configs((camera("CAM-001"),), "CAM-999")


def test_q_requests_clean_display_shutdown() -> None:
    class FakeCV2:
        WND_PROP_VISIBLE = 1

        @staticmethod
        def waitKey(_delay: int) -> int:
            return ord("q")

        @staticmethod
        def getWindowProperty(_name: str, _property: int) -> float:
            raise AssertionError("window visibility is irrelevant after q")

    assert _display_requests_shutdown(FakeCV2()) is True


def test_display_telemetry_distinguishes_renders_from_video_updates() -> None:
    telemetry = DisplayTelemetry("CAM-001")

    telemetry.record(10)
    telemetry.record(10)
    telemetry.record(11)

    assert telemetry.rendered_frames == 3
    assert telemetry.source_frame_updates == 2
    assert telemetry.last_frame_id == 11

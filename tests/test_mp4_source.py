from pathlib import Path

import pytest

from industrial_safety.video import MP4VideoSource, load_camera_configs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config", load_camera_configs(project_root=PROJECT_ROOT))
def test_all_configured_mp4_sources_open_and_decode(config) -> None:
    source = MP4VideoSource(config.source_path)
    try:
        source.open()
        frame = source.read()
        assert source.is_open
        assert source.fps > 0
        assert (source.width, source.height) == (1280, 720)
        assert frame.image.shape[:2] == (720, 1280)
    finally:
        source.release()
    assert not source.is_open


def test_mp4_source_rewinds_and_increments_loop_count() -> None:
    path = PROJECT_ROOT / "videos" / "camera_01_helmet_violation.mp4"
    source = MP4VideoSource(path)
    try:
        source.open()
        assert source.frame_count is not None
        frames = [source.read() for _ in range(source.frame_count + 1)]
    finally:
        source.release()

    assert frames[-2].source_frame_number == source.frame_count - 1
    assert frames[-1].source_frame_number == 0
    assert frames[-1].loop_count == 1

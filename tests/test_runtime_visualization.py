import numpy as np

from industrial_safety.ppe import HelmetTrackAssociator, load_ppe_config
from industrial_safety.runtime import visualization
from industrial_safety.runtime.visualization import (
    EVENT_BANNER_HEIGHT,
    draw_safety_preview,
    find_non_overlapping_label_position,
)
from ppe_helpers import person_track, ppe_result, tracking_result


def test_preview_renders_latest_camera_image_without_ai_results() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    rendered = draw_safety_preview(
        image,
        None,
        camera_fps=30.0,
        person_fps=0.0,
    )

    assert rendered.shape == image.shape
    assert rendered is not image
    assert np.any(rendered != image)


def test_label_is_clamped_inside_frame() -> None:
    rect = find_non_overlapping_label_position(
        (-20, 2, 12, 25),
        (80, 24),
        [],
        (100, 160),
    )

    assert rect[0] >= 0
    assert rect[1] >= 0
    assert rect[2] <= 160
    assert rect[3] <= 100


def test_collision_uses_alternate_position() -> None:
    occupied = [(50, 30, 90, 50)]

    rect = find_non_overlapping_label_position(
        (50, 50, 100, 100),
        (40, 20),
        occupied,
        (180, 240),
    )

    assert rect == (50, 100, 90, 120)


def test_label_avoids_reserved_event_banner() -> None:
    rect = find_non_overlapping_label_position(
        (10, 5, 80, 60),
        (60, 22),
        [(0, 0, 200, EVENT_BANNER_HEIGHT)],
        (120, 200),
        reserved_top=EVENT_BANNER_HEIGHT,
    )

    assert rect[1] >= EVENT_BANNER_HEIGHT


def test_nearby_labels_receive_distinct_rectangles() -> None:
    occupied = []
    for _ in range(4):
        rect = find_non_overlapping_label_position(
            (70, 70, 120, 150),
            (48, 20),
            occupied,
            (240, 320),
        )
        occupied.append(rect)

    assert len(set(occupied)) == 4


def test_unobserved_helmet_is_labeled_missing_in_live_preview(monkeypatch) -> None:
    track = person_track(4, (20, 20, 80, 110))
    tracking = tracking_result((track,))
    association = HelmetTrackAssociator(load_ppe_config().association).associate(
        tracking, ppe_result()
    )
    labels = []

    def capture_label(_canvas, _track, _occupied, **kwargs) -> None:
        labels.append((kwargs["secondary_label"], kwargs["secondary_color"]))

    monkeypatch.setattr(visualization, "draw_person_track", capture_label)
    draw_safety_preview(
        np.zeros((120, 200, 3), dtype=np.uint8),
        tracking,
        ppe_result=association,
    )

    assert labels == [("Helmet Missing", (0, 0, 175))]

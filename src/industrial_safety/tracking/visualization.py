"""Debug-only visualization for camera-local person tracks."""

from __future__ import annotations

import cv2
import numpy as np

from .types import TrackingResult, TrackState


def _track_color(track_id: int, state: TrackState) -> tuple[int, int, int]:
    if state is TrackState.LOST:
        return (96, 96, 96)
    return (
        64 + (track_id * 67) % 192,
        64 + (track_id * 113) % 192,
        64 + (track_id * 157) % 192,
    )


def draw_tracking_result(
    image: np.ndarray,
    result: TrackingResult,
    *,
    inference_fps: float | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for track in result.tracks:
        color = _track_color(track.track_id, track.state)
        point1 = (round(track.x1), round(track.y1))
        point2 = (round(track.x2), round(track.y2))
        thickness = 1 if track.state is TrackState.LOST else 2
        cv2.rectangle(canvas, point1, point2, color, thickness)
        label = f"PERSON #{track.track_id} {track.confidence:.2f}"
        if track.state is not TrackState.CONFIRMED:
            label += f" {track.state.value}"
        cv2.putText(
            canvas,
            label,
            (point1[0], max(20, point1[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            2,
        )
    active = sum(track.state is not TrackState.LOST for track in result.tracks)
    lines = [
        f"{result.camera_id} | frame {result.frame_id}",
        f"active tracks {active} | tracking {result.tracking_ms:.2f} ms",
    ]
    if inference_fps is not None:
        lines.append(f"inference {inference_fps:.2f} FPS")
    if result.tracker_was_reset:
        lines.append("TRACKER RESET")
    for index, line in enumerate(lines, start=1):
        cv2.putText(
            canvas,
            line,
            (10, index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
    return canvas

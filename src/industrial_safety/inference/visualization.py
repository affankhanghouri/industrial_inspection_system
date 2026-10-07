"""Debug-only drawing helpers; no business logic depends on these functions."""

from __future__ import annotations

import cv2
import numpy as np

from .types import InferenceResult


def draw_inference_result(
    image: np.ndarray,
    result: InferenceResult,
    *,
    inference_fps: float | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for detection in result.detections:
        point1 = (round(detection.x1), round(detection.y1))
        point2 = (round(detection.x2), round(detection.y2))
        cv2.rectangle(canvas, point1, point2, (0, 220, 0), 2)
        label = f"PERSON {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (point1[0], max(20, point1[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
        )
    lines = [
        f"{result.camera_id} | frame {result.frame_id}",
        f"persons {len(result.detections)} | inference {result.inference_ms:.0f} ms",
        f"start age {result.frame_age_at_start_ms:.0f} ms | end age {result.end_to_end_age_ms:.0f} ms",
    ]
    if inference_fps is not None:
        lines.append(f"inference {inference_fps:.2f} FPS")
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

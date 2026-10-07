"""Debug visualization for raw and track-associated helmet observations."""

from __future__ import annotations

import cv2
import numpy as np

from .types import PPEAssociationResult, PPEObservationState


def draw_ppe_association(
    image: np.ndarray,
    result: PPEAssociationResult,
) -> np.ndarray:
    canvas = image.copy()
    track_by_id = {
        track.track_id: track for track in result.tracking_result.tracks
    }
    for item in result.track_observations:
        track = track_by_id[item.track_id]
        person_color = (0, 200, 0)
        cv2.rectangle(
            canvas,
            (round(track.x1), round(track.y1)),
            (round(track.x2), round(track.y2)),
            person_color,
            2,
        )
        head = item.head_region
        cv2.rectangle(
            canvas,
            (round(head.x1), round(head.y1)),
            (round(head.x2), round(head.y2)),
            (255, 180, 0),
            1,
        )
        if item.state is PPEObservationState.OBSERVED and item.observation is not None:
            helmet = item.observation
            cv2.rectangle(
                canvas,
                (round(helmet.x1), round(helmet.y1)),
                (round(helmet.x2), round(helmet.y2)),
                (0, 255, 255),
                3,
            )
            label = f"H->#{item.track_id} {helmet.confidence:.2f}"
            label_color = (0, 255, 255)
            label_point = (
                round(helmet.x1),
                max(22, round(helmet.y1) - 8),
            )
            cv2.line(
                canvas,
                (round(helmet.center_x), round(helmet.center_y)),
                (round(head.center_x), round(head.center_y)),
                label_color,
                1,
            )
        else:
            label = f"#{item.track_id} HELMET: UNOBSERVED"
            label_color = (0, 165, 255)
            label_point = (round(track.x1), max(22, round(track.y1) - 8))
        cv2.putText(
            canvas,
            label,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            label_color,
            2,
        )

    for helmet in result.unassociated_observations:
        cv2.rectangle(
            canvas,
            (round(helmet.x1), round(helmet.y1)),
            (round(helmet.x2), round(helmet.y2)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"UNASSOCIATED HELMET {helmet.confidence:.2f}",
            (round(helmet.x1), max(22, round(helmet.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            2,
        )

    observed = sum(
        item.state is PPEObservationState.OBSERVED
        for item in result.track_observations
    )
    cv2.putText(
        canvas,
        (
            f"{result.camera_id} frame {result.frame_id} | raw helmets "
            f"{len(result.ppe_inference_result.observations)} | associated {observed}"
        ),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    return canvas

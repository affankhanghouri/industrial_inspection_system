"""Conservative geometry-only helmet-to-person-track association."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from industrial_safety.tracking.types import PersonTrack, TrackState, TrackingResult

from .config import PPEAssociationConfig
from .types import (
    HeadRegion,
    PPEAssociationResult,
    PPEInferenceResult,
    PPEObservation,
    PPEObservationState,
    PPEType,
    TrackPPEObservation,
)


def helmet_head_region(
    track: PersonTrack,
    config: PPEAssociationConfig,
) -> HeadRegion:
    """Return an expanded top-of-person region for helmet association."""

    horizontal = track.width * config.horizontal_expansion
    vertical = track.height * config.vertical_expansion
    return HeadRegion(
        x1=track.x1 - horizontal,
        y1=track.y1 - vertical,
        x2=track.x2 + horizontal,
        y2=track.y1 + track.height * config.head_region_fraction + vertical,
    )


class HelmetTrackAssociator:
    def __init__(self, config: PPEAssociationConfig) -> None:
        self.config = config

    def associate(
        self,
        tracking: TrackingResult,
        ppe: PPEInferenceResult,
    ) -> PPEAssociationResult:
        if tracking.camera_id != ppe.camera_id:
            raise ValueError("tracking and PPE camera IDs do not match")
        if tracking.frame_id != ppe.frame_id:
            raise ValueError("helmet association requires an exact frame match")
        for observation in ppe.observations:
            if (
                observation.camera_id != ppe.camera_id
                or observation.frame_id != ppe.frame_id
            ):
                raise ValueError("PPE observation identity does not match its result")

        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        eligible_tracks = tuple(
            track
            for track in tracking.tracks
            if track.state is TrackState.CONFIRMED
            and track.source_detection is not None
        )
        regions = {
            track.track_id: helmet_head_region(track, self.config)
            for track in eligible_tracks
        }

        candidates_by_observation: dict[int, list[tuple[float, int]]] = {}
        for observation_index, observation in enumerate(ppe.observations):
            if observation.ppe_type is not PPEType.HELMET:
                continue
            candidates: list[tuple[float, int]] = []
            for track in eligible_tracks:
                score = self._score(regions[track.track_id], observation)
                if score is not None:
                    candidates.append((score, track.track_id))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            if len(candidates) >= 2 and (
                candidates[0][0] - candidates[1][0]
                < self.config.ambiguity_margin
            ):
                # A plausible helmet between similarly scored workers is safer
                # left raw/unassociated than forced onto the wrong trajectory.
                continue
            if candidates:
                candidates_by_observation[observation_index] = candidates

        global_candidates = sorted(
            (
                (score, track_id, observation_index)
                for observation_index, candidates in candidates_by_observation.items()
                for score, track_id in candidates
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        track_matches: dict[int, tuple[int, float]] = {}
        used_observations: set[int] = set()
        for score, track_id, observation_index in global_candidates:
            if track_id in track_matches or observation_index in used_observations:
                continue
            track_matches[track_id] = (observation_index, score)
            used_observations.add(observation_index)

        track_observations: list[TrackPPEObservation] = []
        for track in sorted(eligible_tracks, key=lambda item: item.track_id):
            match = track_matches.get(track.track_id)
            if match is None:
                observation = None
                score = None
                state = PPEObservationState.UNOBSERVED
            else:
                observation_index, score = match
                observation = ppe.observations[observation_index]
                state = PPEObservationState.OBSERVED
            track_observations.append(
                TrackPPEObservation(
                    camera_id=tracking.camera_id,
                    frame_id=tracking.frame_id,
                    track_id=track.track_id,
                    ppe_type=PPEType.HELMET,
                    state=state,
                    head_region=regions[track.track_id],
                    observation=observation,
                    association_score=score,
                )
            )

        finished_monotonic = time.monotonic()
        finished_at = datetime.now(timezone.utc)
        return PPEAssociationResult(
            camera_id=tracking.camera_id,
            frame_id=tracking.frame_id,
            tracking_result=tracking,
            ppe_inference_result=ppe,
            track_observations=tuple(track_observations),
            unassociated_observations=tuple(
                observation
                for index, observation in enumerate(ppe.observations)
                if index not in used_observations
            ),
            association_started_at=started_at,
            association_finished_at=finished_at,
            association_started_monotonic=started_monotonic,
            association_finished_monotonic=finished_monotonic,
            association_ms=max(
                0.0, (finished_monotonic - started_monotonic) * 1000.0
            ),
        )

    def _score(
        self,
        region: HeadRegion,
        observation: PPEObservation,
    ) -> float | None:
        if not (
            region.x1 <= observation.center_x <= region.x2
            and region.y1 <= observation.center_y <= region.y2
        ):
            return None
        intersection_width = max(
            0.0, min(region.x2, observation.x2) - max(region.x1, observation.x1)
        )
        intersection_height = max(
            0.0, min(region.y2, observation.y2) - max(region.y1, observation.y1)
        )
        helmet_area = observation.width * observation.height
        containment = (
            intersection_width * intersection_height / helmet_area
            if helmet_area > 0
            else 0.0
        )
        if containment < self.config.minimum_helmet_overlap:
            return None
        distance = math.hypot(
            observation.center_x - region.center_x,
            observation.center_y - region.center_y,
        )
        diagonal = max(1.0, math.hypot(region.width, region.height))
        normalized_distance = distance / diagonal
        if normalized_distance > self.config.maximum_center_distance:
            return None
        distance_score = max(
            0.0,
            1.0 - normalized_distance / self.config.maximum_center_distance,
        )
        score = 0.70 * min(1.0, containment) + 0.30 * distance_score
        return score if score >= self.config.minimum_score else None

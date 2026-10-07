from dataclasses import replace
from datetime import timedelta

from industrial_safety.ppe import (
    HeadRegion,
    PPEAssociationResult,
    PPEObservationState,
    PPEType,
    TrackPPEObservation,
)
from industrial_safety.rules import Rule, RuleEngine, RuleType
from industrial_safety.zones import Zone, ZoneTransition, ZoneTransitionType

from ppe_helpers import NOW, helmet, person_track, ppe_result, tracking_result


def helmet_rule() -> Rule:
    return Rule(
        "R1", "1", "CAM-1", RuleType.HELMET_REQUIRED, "HELMET_MISSING",
        "high", True, 60.0, minimum_unobserved=2,
        observation_window_seconds=3.0,
    )


def association(frame_id: int, state: PPEObservationState, seconds: float):
    track = person_track(4, (10, 10, 30, 50), frame_id=frame_id)
    tracking = tracking_result((track,), frame_id=frame_id)
    raw = ppe_result(
        (helmet((12, 10, 22, 20), frame_id=frame_id),)
        if state is PPEObservationState.OBSERVED else (),
        frame_id=frame_id,
    )
    raw = replace(
        raw,
        captured_at=NOW + timedelta(seconds=seconds),
        inference_finished_monotonic=seconds,
    )
    observation = raw.observations[0] if raw.observations else None
    return PPEAssociationResult(
        camera_id="CAM-1", frame_id=frame_id, tracking_result=tracking,
        ppe_inference_result=raw,
        track_observations=(TrackPPEObservation(
            "CAM-1", frame_id, 4, PPEType.HELMET, state,
            HeadRegion(8, 8, 32, 25), observation, 0.9 if observation else None,
        ),),
        unassociated_observations=(), association_started_at=NOW,
        association_finished_at=NOW, association_started_monotonic=seconds,
        association_finished_monotonic=seconds + .001, association_ms=1.0,
    )


def test_helmet_temporal_confirmation_requires_repeated_unobserved() -> None:
    engine = RuleEngine((helmet_rule(),))
    assert engine.evaluate_ppe(association(1, PPEObservationState.UNOBSERVED, 1.0)) == ()
    matches = engine.evaluate_ppe(association(2, PPEObservationState.UNOBSERVED, 2.0))
    assert len(matches) == 1
    assert matches[0].rule.event_type == "HELMET_MISSING"
    assert engine.evaluate_ppe(association(3, PPEObservationState.UNOBSERVED, 2.5)) == ()


def test_helmet_observed_cancels_false_confirmation() -> None:
    engine = RuleEngine((helmet_rule(),))
    engine.evaluate_ppe(association(1, PPEObservationState.UNOBSERVED, 1.0))
    engine.evaluate_ppe(association(2, PPEObservationState.OBSERVED, 2.0))
    assert engine.evaluate_ppe(association(3, PPEObservationState.UNOBSERVED, 3.0)) == ()


def test_zone_rule_matches_enter_only() -> None:
    rule = Rule(
        "R2", "1", "CAM-2", RuleType.LOADING_AREA_ENTRY,
        "LOADING_AREA_ENTRY", "high", True, 60.0, zone_id="loading",
    )
    engine = RuleEngine((rule,))
    zone = Zone("loading", "CAM-2", "Loading", "loading_area", ((0,0),(1,0),(1,1)), True, 1, 1)
    track = person_track(2, (0, 0, 1, 1), camera_id="CAM-2")
    enter = ZoneTransition("CAM-2", 1, zone, track, ZoneTransitionType.ENTER, NOW)
    exit_transition = replace(enter, transition=ZoneTransitionType.EXIT)
    assert len(engine.evaluate_zone(enter)) == 1
    assert engine.evaluate_zone(exit_transition) == ()

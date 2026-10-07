"""Three-scenario rule matching with bounded helmet temporal evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from industrial_safety.ppe.types import PPEAssociationResult, PPEObservationState
from industrial_safety.zones.types import ZoneTransition, ZoneTransitionType

from .types import Rule, RuleMatch, RuleType


@dataclass(slots=True)
class _HelmetState:
    unobserved_at: deque[float] = field(default_factory=lambda: deque(maxlen=16))
    active: bool = False


class RuleEngine:
    def __init__(self, rules: tuple[Rule, ...]) -> None:
        self.rules = tuple(rule for rule in rules if rule.enabled)
        self._helmet_states: dict[tuple[str, str, int], _HelmetState] = {}

    def reset_camera(self, camera_id: str) -> None:
        self._helmet_states = {
            key: state
            for key, state in self._helmet_states.items()
            if key[0] != camera_id
        }

    def evaluate_ppe(self, result: PPEAssociationResult) -> tuple[RuleMatch, ...]:
        if result.tracking_result.tracker_was_reset:
            self.reset_camera(result.camera_id)
        matches: list[RuleMatch] = []
        rules = [
            rule
            for rule in self.rules
            if rule.camera_id == result.camera_id
            and rule.rule_type is RuleType.HELMET_REQUIRED
        ]
        observed = result.ppe_inference_result.inference_finished_monotonic
        for rule in rules:
            assert rule.minimum_unobserved is not None
            assert rule.observation_window_seconds is not None
            for item in result.track_observations:
                key = (result.camera_id, rule.rule_id, item.track_id)
                state = self._helmet_states.setdefault(key, _HelmetState())
                if item.state is PPEObservationState.OBSERVED:
                    state.unobserved_at.clear()
                    state.active = False
                    continue
                cutoff = observed - rule.observation_window_seconds
                while state.unobserved_at and state.unobserved_at[0] < cutoff:
                    state.unobserved_at.popleft()
                state.unobserved_at.append(observed)
                if state.active or len(state.unobserved_at) < rule.minimum_unobserved:
                    continue
                track = next(
                    track
                    for track in result.tracking_result.tracks
                    if track.track_id == item.track_id
                )
                state.active = True
                matches.append(
                    RuleMatch(
                        rule=rule,
                        camera_id=result.camera_id,
                        track=track,
                        detected_at=result.ppe_inference_result.captured_at,
                        zone_id=None,
                        confidence=None,
                        supporting_observations=len(state.unobserved_at),
                        ppe_result=result,
                    )
                )
        return tuple(matches)

    def evaluate_zone(self, transition: ZoneTransition) -> tuple[RuleMatch, ...]:
        if transition.transition is not ZoneTransitionType.ENTER:
            return ()
        matches = []
        for rule in self.rules:
            if rule.camera_id != transition.camera_id or rule.zone_id != transition.zone.zone_id:
                continue
            if rule.rule_type not in {
                RuleType.LOADING_AREA_ENTRY,
                RuleType.RESTRICTED_AREA_ENTRY,
            }:
                continue
            matches.append(
                RuleMatch(
                    rule=rule,
                    camera_id=transition.camera_id,
                    track=transition.track,
                    detected_at=transition.occurred_at,
                    zone_id=transition.zone.zone_id,
                    confidence=transition.track.confidence,
                    supporting_observations=1,
                    zone_transition=transition,
                )
            )
        return tuple(matches)

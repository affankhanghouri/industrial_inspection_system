from dataclasses import replace

from industrial_safety.zones import (
    Zone,
    ZoneEngine,
    ZoneState,
    ZoneTransitionType,
    point_in_polygon,
)

from ppe_helpers import person_track, tracking_result


POLYGON = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))


def zone(confirmations: int = 2) -> Zone:
    return Zone(
        zone_id="test-zone",
        camera_id="CAM-1",
        name="Test",
        zone_type="restricted_area",
        polygon=POLYGON,
        enabled=True,
        enter_confirmations=confirmations,
        exit_confirmations=confirmations,
    )


def result(frame_id: int, bottom_center: tuple[float, float], *, reset=False):
    x, y = bottom_center
    track = person_track(1, (x - 2, y - 10, x + 2, y), frame_id=frame_id)
    base = tracking_result((track,), frame_id=frame_id)
    return replace(base, tracker_was_reset=reset, reset_reason="loop" if reset else None)


def test_point_in_polygon_inside_outside_and_edge() -> None:
    assert point_in_polygon((20, 20), POLYGON)
    assert point_in_polygon((10, 20), POLYGON)
    assert not point_in_polygon((5, 20), POLYGON)


def test_zone_enter_exit_and_continuous_inside_are_debounced() -> None:
    engine = ZoneEngine((zone(),))
    assert engine.evaluate(result(1, (5, 20))).transitions == ()
    assert engine.evaluate(result(2, (20, 20))).transitions == ()
    entered = engine.evaluate(result(3, (20, 20)))
    assert entered.transitions[0].transition is ZoneTransitionType.ENTER
    assert entered.observations[0].state is ZoneState.INSIDE
    assert engine.evaluate(result(4, (20, 20))).transitions == ()
    assert engine.evaluate(result(5, (5, 20))).transitions == ()
    exited = engine.evaluate(result(6, (5, 20)))
    assert exited.transitions[0].transition is ZoneTransitionType.EXIT


def test_edge_jitter_does_not_create_transition() -> None:
    engine = ZoneEngine((zone(confirmations=2),))
    engine.evaluate(result(1, (5, 20)))
    engine.evaluate(result(2, (20, 20)))
    evaluation = engine.evaluate(result(3, (5, 20)))
    assert evaluation.transitions == ()
    assert evaluation.observations[0].state is ZoneState.OUTSIDE


def test_tracker_reset_clears_zone_state_without_synthetic_entry() -> None:
    engine = ZoneEngine((zone(confirmations=1),))
    engine.evaluate(result(1, (5, 20)))
    assert engine.evaluate(result(2, (20, 20))).transitions
    after_reset = engine.evaluate(result(3, (20, 20), reset=True))
    assert after_reset.reset_applied
    assert after_reset.transitions == ()

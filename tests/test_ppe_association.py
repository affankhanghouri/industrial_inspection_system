from industrial_safety.ppe import (
    HelmetTrackAssociator,
    PPEObservationState,
    TrackPPEObservation,
    helmet_head_region,
    load_ppe_config,
)
from industrial_safety.tracking import TrackState

from ppe_helpers import helmet, person_track, ppe_result, tracking_result


def associator() -> HelmetTrackAssociator:
    return HelmetTrackAssociator(load_ppe_config().association)


def test_head_region_is_expanded_top_portion_of_person() -> None:
    track = person_track(1, (100, 50, 200, 250))
    region = helmet_head_region(track, load_ppe_config().association)

    assert region.bbox == (80.0, 40.0, 220.0, 120.0)


def test_helmet_inside_correct_head_region_associates() -> None:
    track = person_track(1, (0, 0, 100, 200))
    observation = helmet((30, 5, 70, 45))

    result = associator().associate(
        tracking_result((track,)), ppe_result((observation,))
    )

    assert result.track_observations[0].state is PPEObservationState.OBSERVED
    assert result.track_observations[0].observation is observation
    assert result.track_observations[0].association_score is not None
    assert result.unassociated_observations == ()


def test_nearby_workers_receive_independent_helmet_associations() -> None:
    first = person_track(1, (0, 0, 100, 200))
    second = person_track(2, (110, 0, 210, 200))
    first_helmet = helmet((25, 5, 70, 45), confidence=0.91)
    second_helmet = helmet((135, 5, 180, 45), confidence=0.88)

    result = associator().associate(
        tracking_result((first, second)),
        ppe_result((first_helmet, second_helmet)),
    )

    assert [item.observation for item in result.track_observations] == [
        first_helmet,
        second_helmet,
    ]


def test_one_helmet_cannot_attach_to_two_tracks() -> None:
    first = person_track(1, (0, 0, 100, 200))
    second = person_track(2, (120, 0, 220, 200))
    observation = helmet((25, 5, 70, 45))

    result = associator().associate(
        tracking_result((first, second)), ppe_result((observation,))
    )

    assert sum(
        item.state is PPEObservationState.OBSERVED
        for item in result.track_observations
    ) == 1
    assert sum(
        item.state is PPEObservationState.UNOBSERVED
        for item in result.track_observations
    ) == 1


def test_ambiguous_helmet_remains_unassociated() -> None:
    first = person_track(1, (0, 0, 100, 200))
    second = person_track(2, (0, 0, 100, 200))
    observation = helmet((30, 5, 70, 45))

    result = associator().associate(
        tracking_result((first, second)), ppe_result((observation,))
    )

    assert all(
        item.state is PPEObservationState.UNOBSERVED
        for item in result.track_observations
    )
    assert result.unassociated_observations == (observation,)


def test_no_matched_helmet_is_unobserved_never_violation() -> None:
    track = person_track(1, (0, 0, 100, 200))

    result = associator().associate(tracking_result((track,)), ppe_result())
    item: TrackPPEObservation = result.track_observations[0]

    assert item.state is PPEObservationState.UNOBSERVED
    assert item.confidence is None
    assert item.ppe_bbox is None
    assert "VIOLATION" not in {state.value for state in PPEObservationState}


def test_lost_predicted_track_is_not_evaluated_as_fresh_evidence() -> None:
    lost = person_track(
        1,
        (0, 0, 100, 200),
        state=TrackState.LOST,
        fresh=False,
    )

    result = associator().associate(
        tracking_result((lost,)), ppe_result((helmet((30, 5, 70, 45)),))
    )

    assert result.track_observations == ()
    assert len(result.unassociated_observations) == 1


def test_association_handles_zero_people() -> None:
    observation = helmet((30, 5, 70, 45))
    result = associator().associate(
        tracking_result(()), ppe_result((observation,))
    )

    assert result.track_observations == ()
    assert result.unassociated_observations == (observation,)


def test_association_handles_zero_helmets() -> None:
    first = person_track(1, (0, 0, 100, 200))
    second = person_track(2, (120, 0, 220, 200))
    result = associator().associate(
        tracking_result((first, second)), ppe_result()
    )

    assert [item.state for item in result.track_observations] == [
        PPEObservationState.UNOBSERVED,
        PPEObservationState.UNOBSERVED,
    ]

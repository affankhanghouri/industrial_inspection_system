from dataclasses import replace

from industrial_safety.tracking import (
    PersonTrack,
    PersonTracker,
    TimeAwareIoUPersonTracker,
    TrackState,
    load_tracking_config,
)

from tracking_helpers import inference_result


def tracker(camera_id: str = "CAM-1") -> TimeAwareIoUPersonTracker:
    config = load_tracking_config().tracker
    return TimeAwareIoUPersonTracker(camera_id, config)


def test_tracker_interface_returns_structured_tracks() -> None:
    person_tracker: PersonTracker = tracker()
    tracks = person_tracker.update(
        inference_result("CAM-1", 0, 0.0, ((0, 0, 20, 40),))
    )

    assert isinstance(tracks, tuple)
    assert isinstance(tracks[0], PersonTrack)
    assert tracks[0].track_id == 1
    assert tracks[0].bottom_center == (10.0, 40.0)


def test_each_camera_has_independent_track_ids() -> None:
    first = tracker("CAM-1")
    second = tracker("CAM-2")

    assert first.update(inference_result("CAM-1", 0, 0.0, ((0, 0, 20, 40),)))[0].track_id == 1
    assert second.update(inference_result("CAM-2", 0, 0.0, ((100, 0, 120, 40),)))[0].track_id == 1


def test_moving_person_keeps_stable_id_with_frame_gaps() -> None:
    person_tracker = tracker()
    ids = []
    for frame_id, observed, x1 in ((0, 0.0, 0), (16, 0.5, 5), (34, 1.0, 11)):
        tracks = person_tracker.update(
            inference_result(
                "CAM-1", frame_id, observed, ((x1, 0, x1 + 20, 40),)
            )
        )
        ids.append(tracks[0].track_id)

    assert ids == [1, 1, 1]
    assert tracks[0].state is TrackState.CONFIRMED
    assert tracks[0].hits == 3


def test_two_people_receive_different_ids() -> None:
    tracks = tracker().update(
        inference_result(
            "CAM-1", 0, 0.0, ((0, 0, 20, 40), (100, 0, 120, 40))
        )
    )

    assert {track.track_id for track in tracks} == {1, 2}


def test_brief_miss_preserves_confirmed_track() -> None:
    person_tracker = tracker()
    person_tracker.update(inference_result("CAM-1", 0, 0.0, ((0, 0, 20, 40),)))
    person_tracker.update(inference_result("CAM-1", 15, 0.5, ((4, 0, 24, 40),)))

    missed = person_tracker.update(inference_result("CAM-1", 30, 1.0))
    recovered = person_tracker.update(
        inference_result("CAM-1", 45, 1.5, ((12, 0, 32, 40),))
    )

    assert missed[0].track_id == 1
    assert missed[0].state is TrackState.LOST
    assert recovered[0].track_id == 1
    assert recovered[0].state is TrackState.CONFIRMED


def test_track_expires_by_elapsed_time() -> None:
    config = replace(load_tracking_config().tracker, maximum_lost_seconds=1.0)
    person_tracker = TimeAwareIoUPersonTracker("CAM-1", config)
    person_tracker.update(inference_result("CAM-1", 0, 0.0, ((0, 0, 20, 40),)))

    assert person_tracker.update(inference_result("CAM-1", 15, 0.5))[0].state is TrackState.LOST
    assert person_tracker.update(inference_result("CAM-1", 60, 2.0)) == ()


def test_no_detection_input_is_valid() -> None:
    assert tracker().update(inference_result("CAM-1", 50, 2.0)) == ()

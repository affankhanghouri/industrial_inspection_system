from industrial_safety.ppe import (
    HelmetTrackAssociator,
    LatestPPEAssociationResult,
    LatestPPEInferenceResult,
    load_ppe_config,
)

from ppe_helpers import person_track, ppe_result, tracking_result


def test_latest_ppe_results_overwrite_stale_results() -> None:
    inference_buffer = LatestPPEInferenceResult()
    first = ppe_result(frame_id=10)
    second = ppe_result(frame_id=20)
    inference_buffer.publish(first)
    inference_buffer.publish(second)

    assert inference_buffer.stored_result_count == 1
    assert inference_buffer.get_latest(timeout=0).frame_id == 20

    association_buffer = LatestPPEAssociationResult()
    associator = HelmetTrackAssociator(load_ppe_config().association)
    first_association = associator.associate(
        tracking_result(
            (person_track(1, (0, 0, 100, 200), frame_id=10),),
            frame_id=10,
        ),
        first,
    )
    second_association = associator.associate(
        tracking_result(
            (person_track(1, (0, 0, 100, 200), frame_id=20),),
            frame_id=20,
        ),
        second,
    )
    association_buffer.publish(first_association)
    association_buffer.publish(second_association)

    assert association_buffer.stored_result_count == 1
    assert association_buffer.get_latest(timeout=0).frame_id == 20

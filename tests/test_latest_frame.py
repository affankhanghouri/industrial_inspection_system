from datetime import datetime, timezone

from industrial_safety.video import FramePacket, LatestFrameBuffer


def packet(frame_id: int) -> FramePacket:
    return FramePacket(
        camera_id="CAM-TEST",
        frame_id=frame_id,
        image=object(),
        captured_at=datetime.now(timezone.utc),
        captured_monotonic=1.0,
        source_timestamp_ms=float(frame_id),
        source_frame_number=frame_id,
        loop_count=0,
        width=1,
        height=1,
    )


def test_latest_frame_overwrites_stale_frames() -> None:
    buffer = LatestFrameBuffer()
    for frame_id in range(100, 105):
        buffer.publish(packet(frame_id))

    assert buffer.stored_frame_count == 1
    assert buffer.get_latest(timeout=0).frame_id == 104


def test_consumer_can_request_only_a_newer_frame() -> None:
    buffer = LatestFrameBuffer()
    buffer.publish(packet(7))

    assert buffer.get_latest(after_frame_id=6, timeout=0).frame_id == 7
    assert buffer.get_latest(after_frame_id=7, timeout=0.01) is None

    buffer.publish(packet(8))
    assert buffer.get_latest(after_frame_id=7, timeout=0).frame_id == 8

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from industrial_safety.events import (
    EvidenceConfig,
    EvidenceRecorder,
    EvidenceRingBuffer,
)
from industrial_safety.events.evidence import EvidenceFrame
from industrial_safety.events.service import EventService
from industrial_safety.storage import SQLiteEventStore

from inference_helpers import frame
from test_events_api import rule_match


def packet(frame_id: int, monotonic: float):
    return replace(
        frame("CAM-2", frame_id),
        captured_monotonic=monotonic,
        image=np.full((32, 48, 3), frame_id * 10, dtype=np.uint8),
        width=48,
        height=32,
    )


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_evidence_ring_buffer_is_bounded_and_retains_newest_frames() -> None:
    buffer = EvidenceRingBuffer(3)
    for frame_id in range(5):
        buffer.append(EvidenceFrame(frame_id, float(frame_id), np.zeros((2, 2, 3))))

    assert len(buffer) == 3
    assert [sample.frame_id for sample in buffer.snapshot()] == [2, 3, 4]


def test_pre_and_post_event_frames_are_written_to_one_clip(tmp_path) -> None:
    config = EvidenceConfig(fps=10, pre_seconds=0.3, post_seconds=0.2, max_width=48)
    recorder = EvidenceRecorder(("CAM-2",), lambda _camera_id: None, tmp_path, config=config)
    recorder.start()
    for frame_id in range(3):
        recorder.record_frame(packet(frame_id, frame_id / 10))
    assert recorder.buffer_size("CAM-2") == 3
    assert recorder.trigger("event-1", "CAM-2", "day/event-1/clip.mp4", triggered_monotonic=0.2)
    recorder.record_frame(packet(3, 0.3))
    recorder.record_frame(packet(4, 0.4))
    assert wait_for(lambda: recorder.stats()["clips_written"] == 1)
    recorder.stop()

    clip = tmp_path / "day/event-1/clip.mp4"
    capture = cv2.VideoCapture(str(clip))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()


def test_duplicate_capture_trigger_does_not_generate_duplicate_clip(tmp_path) -> None:
    config = EvidenceConfig(fps=10, pre_seconds=0.1, post_seconds=0.1, max_width=48)
    recorder = EvidenceRecorder(("CAM-2",), lambda _camera_id: None, tmp_path, config=config)
    recorder.start()
    recorder.record_frame(packet(1, 0.0))
    assert recorder.trigger("confirmed", "CAM-2", "day/confirmed/clip.mp4", triggered_monotonic=0.0)
    assert not recorder.trigger("confirmed", "CAM-2", "day/confirmed/second.mp4", triggered_monotonic=0.0)
    recorder.record_frame(packet(2, 0.1))
    assert wait_for(lambda: recorder.stats()["clips_written"] == 1)
    recorder.stop()

    assert (tmp_path / "day/confirmed/clip.mp4").is_file()
    assert not (tmp_path / "day/confirmed/second.mp4").exists()


class TriggerRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def trigger(self, event_id, camera_id, relative_clip_path, **_kwargs) -> bool:
        self.calls.append((event_id, camera_id, str(relative_clip_path)))
        return True


def test_confirmed_event_persists_snapshot_and_clip_paths_once(tmp_path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    recorder = TriggerRecorder()
    service = EventService(
        store,
        tmp_path / "evidence",
        person_model_version="person",
        ppe_model_version="ppe",
        evidence_recorder=recorder,  # type: ignore[arg-type]
    )

    event = service.create(rule_match(), frame("CAM-2", 10))
    assert event is not None
    assert event.snapshot_path.endswith(f"/{event.event_id}/snapshot.jpg")
    assert event.clip_path == event.snapshot_path.replace("snapshot.jpg", "clip.mp4")
    assert store.get(event.event_id).clip_path == event.clip_path  # type: ignore[union-attr]
    assert len(recorder.calls) == 1

    duplicate = service.create(rule_match(), frame("CAM-2", 11))
    assert duplicate is None
    assert len(recorder.calls) == 1
    store.close()


def test_event_without_recorder_has_snapshot_and_no_clip(tmp_path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    service = EventService(
        store, tmp_path / "evidence", person_model_version="person", ppe_model_version="ppe"
    )
    event = service.create(rule_match(), frame("CAM-2", 10))

    assert event is not None
    assert (tmp_path / "evidence" / event.snapshot_path).is_file()
    assert event.clip_path is None
    store.close()

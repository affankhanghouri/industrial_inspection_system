#!/usr/bin/env python3
"""Generate CAM-001 helmet association images and diagnostic history."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.inference import (
    InferenceResult,
    UltralyticsPersonDetector,
    load_inference_config,
)
from industrial_safety.ppe import (
    HelmetTrackAssociator,
    PPEInferenceResult,
    PPEObservationState,
    UltralyticsPPEDetector,
    load_ppe_config,
)
from industrial_safety.ppe.visualization import draw_ppe_association
from industrial_safety.tracking import (
    CameraTrackingProcessor,
    TimeAwareIoUPersonTracker,
    load_tracking_config,
)
from industrial_safety.video import FramePacket, load_camera_configs


def main() -> int:
    output_dir = PROJECT_ROOT / "debug" / "ppe"
    output_dir.mkdir(parents=True, exist_ok=True)
    person_detector = UltralyticsPersonDetector(
        load_inference_config(project_root=PROJECT_ROOT).detector
    )
    ppe_config = load_ppe_config(project_root=PROJECT_ROOT)
    ppe_detector = UltralyticsPPEDetector(ppe_config.detector)
    person_detector.warmup()
    ppe_detector.warmup()
    tracker = CameraTrackingProcessor(
        "CAM-001",
        TimeAwareIoUPersonTracker(
            "CAM-001", load_tracking_config(project_root=PROJECT_ROOT).tracker
        ),
    )
    associator = HelmetTrackAssociator(ppe_config.association)
    camera = next(
        item
        for item in load_camera_configs(project_root=PROJECT_ROOT)
        if item.camera_id == "CAM-001"
    )
    capture = cv2.VideoCapture(str(camera.source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {camera.source_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    sample_step = max(1, round(fps / 2.0))
    sample_frames = list(range(0, frame_count, sample_step))
    selected_indices = {1, 5, 9, 12, len(sample_frames) - 1}
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [
        "CAM-001 helmet perception history",
        "UNOBSERVED means only that no helmet was confidently observed.",
        "",
    ]
    tiles: list[np.ndarray] = []

    for observation_index, source_frame in enumerate(sample_frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = capture.read()
        if not ok or image is None:
            continue
        observed = 1000.0 + source_frame / fps
        captured_at = base_time + timedelta(seconds=source_frame / fps)
        packet = FramePacket(
            camera_id="CAM-001",
            frame_id=source_frame,
            image=image,
            captured_at=captured_at,
            captured_monotonic=observed,
            source_timestamp_ms=source_frame * 1000.0 / fps,
            source_frame_number=source_frame,
            loop_count=0,
            width=image.shape[1],
            height=image.shape[0],
        )

        person_started = time.perf_counter()
        people = person_detector.detect(packet)
        person_ms = (time.perf_counter() - person_started) * 1000.0
        inference = InferenceResult(
            camera_id="CAM-001",
            frame_id=source_frame,
            captured_at=captured_at,
            source_timestamp_ms=packet.source_timestamp_ms,
            source_frame_number=source_frame,
            source_loop_count=0,
            inference_started_at=captured_at,
            inference_finished_at=captured_at + timedelta(milliseconds=person_ms),
            inference_started_monotonic=observed,
            inference_finished_monotonic=observed + person_ms / 1000.0,
            detections=people,
            inference_ms=person_ms,
            frame_age_at_start_ms=0.0,
            end_to_end_age_ms=person_ms,
        )
        tracking = tracker.process(inference)

        ppe_started_monotonic = time.monotonic()
        ppe_started_at = datetime.now(timezone.utc)
        ppe_observations = ppe_detector.detect(packet)
        ppe_finished_monotonic = time.monotonic()
        ppe_finished_at = datetime.now(timezone.utc)
        ppe_ms = (ppe_finished_monotonic - ppe_started_monotonic) * 1000.0
        ppe_result = PPEInferenceResult(
            camera_id="CAM-001",
            frame_id=source_frame,
            captured_at=captured_at,
            source_timestamp_ms=packet.source_timestamp_ms,
            source_frame_number=source_frame,
            source_loop_count=0,
            observations=ppe_observations,
            inference_started_at=ppe_started_at,
            inference_finished_at=ppe_finished_at,
            inference_started_monotonic=ppe_started_monotonic,
            inference_finished_monotonic=ppe_finished_monotonic,
            inference_ms=ppe_ms,
            frame_age_at_start_ms=0.0,
            end_to_end_age_ms=ppe_ms,
        )
        association = associator.associate(tracking, ppe_result)
        history.append(
            f"t={source_frame / fps:4.1f}s frame={source_frame:3d} "
            f"people={len(people)} raw_helmets={len(ppe_observations)} "
            f"unassociated={len(association.unassociated_observations)}"
        )
        for item in association.track_observations:
            suffix = (
                f" confidence={item.confidence:.3f} score={item.association_score:.3f}"
                if item.state is PPEObservationState.OBSERVED
                else ""
            )
            history.append(
                f"  track={item.track_id:2d} helmet={item.state.value}{suffix}"
            )
        if observation_index in selected_indices:
            annotated = draw_ppe_association(image, association)
            image_path = output_dir / f"cam-001_obs_{observation_index:02d}.jpg"
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"failed to write {image_path}")
            tiles.append(cv2.resize(annotated, (480, 270)))

    capture.release()
    tracker.close()
    history_path = output_dir / "helmet_history.txt"
    history_path.write_text("\n".join(history), encoding="utf-8")
    sequence_path = output_dir / "cam-001_sequence.jpg"
    if not cv2.imwrite(str(sequence_path), np.hstack(tiles)):
        raise RuntimeError(f"failed to write {sequence_path}")
    print(f"observations={len(sample_frames)} history={history_path}")
    print(f"sequence={sequence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate annotated tracking sequences and a textual continuity history."""

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
from industrial_safety.tracking import (
    CameraTrackingProcessor,
    TimeAwareIoUPersonTracker,
    load_tracking_config,
)
from industrial_safety.tracking.visualization import draw_tracking_result
from industrial_safety.video import FramePacket, load_camera_configs


def build_inference_result(
    detector: UltralyticsPersonDetector,
    *,
    camera_id: str,
    frame_id: int,
    source_frame_number: int,
    source_timestamp_ms: float,
    source_loop_count: int,
    image: np.ndarray,
    base_wall_time: datetime,
) -> InferenceResult:
    observed_monotonic = 1000.0 + source_loop_count * 100.0 + source_timestamp_ms / 1000.0
    captured_at = base_wall_time + timedelta(
        seconds=source_loop_count * 100.0 + source_timestamp_ms / 1000.0
    )
    packet = FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        image=image,
        captured_at=captured_at,
        captured_monotonic=observed_monotonic,
        source_timestamp_ms=source_timestamp_ms,
        source_frame_number=source_frame_number,
        loop_count=source_loop_count,
        width=image.shape[1],
        height=image.shape[0],
    )
    started = time.perf_counter()
    detections = detector.detect(packet)
    inference_ms = (time.perf_counter() - started) * 1000.0
    return InferenceResult(
        camera_id=camera_id,
        frame_id=frame_id,
        captured_at=captured_at,
        source_timestamp_ms=source_timestamp_ms,
        source_frame_number=source_frame_number,
        source_loop_count=source_loop_count,
        inference_started_at=captured_at + timedelta(milliseconds=10),
        inference_finished_at=captured_at + timedelta(milliseconds=10 + inference_ms),
        inference_started_monotonic=observed_monotonic + 0.010,
        inference_finished_monotonic=observed_monotonic + 0.010 + inference_ms / 1000.0,
        detections=detections,
        inference_ms=inference_ms,
        frame_age_at_start_ms=10.0,
        end_to_end_age_ms=10.0 + inference_ms,
    )


def main() -> int:
    output_dir = PROJECT_ROOT / "debug" / "tracking"
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = UltralyticsPersonDetector(
        load_inference_config(project_root=PROJECT_ROOT).detector
    )
    detector.warmup()
    tracker_config = load_tracking_config(project_root=PROJECT_ROOT).tracker
    history: list[str] = []
    base_wall_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for camera in load_camera_configs(project_root=PROJECT_ROOT):
        if not camera.enabled:
            continue
        capture = cv2.VideoCapture(str(camera.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {camera.source_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        sample_step = max(1, round(fps / 2.0))
        sample_frames = list(range(0, frame_count, sample_step))
        selected_indices = {
            round(index * (len(sample_frames) - 1) / 3) for index in range(4)
        }
        # Keep the 6-second observation as a focused review point. CAM-001 has a
        # short-lived overlapping detector box here, and the same timestamp is
        # useful for comparing motion/occlusion behavior across all cameras.
        if 180 in sample_frames:
            selected_indices.add(sample_frames.index(180))
        processor = CameraTrackingProcessor(
            camera.camera_id,
            TimeAwareIoUPersonTracker(camera.camera_id, tracker_config),
        )
        history.extend([camera.camera_id, "=" * len(camera.camera_id)])
        sequence_tiles: list[np.ndarray] = []

        for observation_index, source_frame in enumerate(sample_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            source_timestamp_ms = source_frame * 1000.0 / fps
            inference = build_inference_result(
                detector,
                camera_id=camera.camera_id,
                frame_id=source_frame,
                source_frame_number=source_frame,
                source_timestamp_ms=source_timestamp_ms,
                source_loop_count=0,
                image=image,
                base_wall_time=base_wall_time,
            )
            tracking = processor.process(inference)
            history.append(
                f"t={source_timestamp_ms / 1000.0:5.2f}s frame={source_frame:3d} "
                f"detections={len(inference.detections)} tracks={len(tracking.tracks)}"
            )
            for track in tracking.tracks:
                history.append(
                    f"  track={track.track_id:2d} state={track.state.value:9s} "
                    f"hits={track.hits:2d} missed={track.missed_updates:2d} "
                    f"bbox=({track.x1:.0f},{track.y1:.0f},{track.x2:.0f},{track.y2:.0f})"
                )
            if observation_index in selected_indices:
                annotated = draw_tracking_result(image, tracking)
                image_path = output_dir / (
                    f"{camera.camera_id.lower()}_obs_{observation_index:02d}.jpg"
                )
                if not cv2.imwrite(str(image_path), annotated):
                    raise RuntimeError(f"failed to write {image_path}")
                sequence_tiles.append(cv2.resize(annotated, (480, 270)))

        # Process the first source frame as the next loop to validate reset behavior.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, loop_image = capture.read()
        capture.release()
        if not ok or loop_image is None:
            raise RuntimeError(f"could not decode loop frame for {camera.camera_id}")
        loop_inference = build_inference_result(
            detector,
            camera_id=camera.camera_id,
            frame_id=frame_count,
            source_frame_number=0,
            source_timestamp_ms=0.0,
            source_loop_count=1,
            image=loop_image,
            base_wall_time=base_wall_time,
        )
        loop_result = processor.process(loop_inference)
        history.append(
            f"LOOP frame={frame_count} reset={loop_result.tracker_was_reset} "
            f"reason={loop_result.reset_reason}"
        )
        history.append("")
        processor.close()

        contact_sheet = np.hstack(sequence_tiles)
        sheet_path = output_dir / f"{camera.camera_id.lower()}_sequence.jpg"
        if not cv2.imwrite(str(sheet_path), contact_sheet):
            raise RuntimeError(f"failed to write {sheet_path}")
        print(
            f"{camera.camera_id}: observations={len(sample_frames)} "
            f"sequence={sheet_path} loop_reset={loop_result.tracker_was_reset}"
        )

    history_path = output_dir / "track_history.txt"
    history_path.write_text("\n".join(history), encoding="utf-8")
    print(f"history={history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

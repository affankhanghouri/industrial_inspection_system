#!/usr/bin/env python3
"""Save one representative person-detection debug image per camera."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.inference import (
    InferenceResult,
    UltralyticsPersonDetector,
    load_inference_config,
)
from industrial_safety.inference.visualization import draw_inference_result
from industrial_safety.video import FramePacket, load_camera_configs


def packet_for_frame(camera_id: str, frame_id: int, image, source_ms: float) -> FramePacket:
    now = time.monotonic()
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        image=image,
        captured_at=datetime.now(timezone.utc),
        captured_monotonic=now,
        source_timestamp_ms=source_ms,
        source_frame_number=frame_id,
        loop_count=0,
        width=image.shape[1],
        height=image.shape[0],
    )


def main() -> int:
    output_dir = PROJECT_ROOT / "debug" / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = load_inference_config(project_root=PROJECT_ROOT)
    detector = UltralyticsPersonDetector(inference.detector)
    detector.warmup()

    for camera in load_camera_configs(project_root=PROJECT_ROOT):
        if not camera.enabled:
            continue
        capture = cv2.VideoCapture(str(camera.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {camera.source_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        candidates = sorted({round(index * (frame_count - 1) / 11) for index in range(12)})
        best = None
        for frame_id in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            packet = packet_for_frame(
                camera.camera_id,
                frame_id,
                image,
                float(capture.get(cv2.CAP_PROP_POS_MSEC)),
            )
            started_mono = time.monotonic()
            started_at = datetime.now(timezone.utc)
            detections = detector.detect(packet)
            finished_mono = time.monotonic()
            finished_at = datetime.now(timezone.utc)
            confidence_sum = sum(detection.confidence for detection in detections)
            candidate = (len(detections), confidence_sum, packet, detections, started_mono, started_at, finished_mono, finished_at)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        capture.release()
        if best is None:
            raise RuntimeError(f"no decodable candidate frame for {camera.camera_id}")
        count, _, packet, detections, started_mono, started_at, finished_mono, finished_at = best
        result = InferenceResult(
            camera_id=camera.camera_id,
            frame_id=packet.frame_id,
            captured_at=packet.captured_at,
            source_timestamp_ms=packet.source_timestamp_ms,
            source_frame_number=packet.source_frame_number,
            source_loop_count=packet.loop_count,
            inference_started_at=started_at,
            inference_finished_at=finished_at,
            inference_started_monotonic=started_mono,
            inference_finished_monotonic=finished_mono,
            detections=detections,
            inference_ms=(finished_mono - started_mono) * 1000.0,
            frame_age_at_start_ms=(started_mono - packet.captured_monotonic) * 1000.0,
            end_to_end_age_ms=(finished_mono - packet.captured_monotonic) * 1000.0,
        )
        annotated = draw_inference_result(packet.image, result)
        output_path = output_dir / f"{camera.camera_id.lower()}_persons.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"failed to write {output_path}")
        print(
            f"{camera.camera_id}: frame={packet.frame_id} persons={count} "
            f"inference_ms={result.inference_ms:.1f} output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

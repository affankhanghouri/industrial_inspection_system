#!/usr/bin/env python3
"""Benchmark person detection on representative frames from configured videos."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.inference import create_person_detector, load_inference_config
from industrial_safety.video import FramePacket, load_camera_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--backend", choices=("ultralytics", "onnxruntime", "openvino"))
    parser.add_argument("--model", type=Path, default=None)
    return parser.parse_args()


def representative_frame(camera_id: str, path: Path) -> FramePacket:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, image = capture.read()
    source_position = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"could not decode representative frame from {path}")
    now = time.monotonic()
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_count // 2,
        image=image,
        captured_at=datetime.now(timezone.utc),
        captured_monotonic=now,
        source_timestamp_ms=source_position,
        source_frame_number=frame_count // 2,
        loop_count=0,
        width=image.shape[1],
        height=image.shape[0],
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999) - 1))]


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise SystemExit("iterations must be positive and warmup cannot be negative")
    inference = load_inference_config(project_root=PROJECT_ROOT)
    detector_config = inference.detector
    if args.threads is not None:
        detector_config = replace(detector_config, cpu_threads=args.threads)
    if args.imgsz is not None:
        detector_config = replace(detector_config, image_size=args.imgsz)
    if args.confidence is not None:
        detector_config = replace(detector_config, confidence=args.confidence)
    if args.backend is not None:
        detector_config = replace(detector_config, backend=args.backend)
    if args.model is not None:
        detector_config = replace(detector_config, model_path=args.model.resolve())
    detector = create_person_detector(detector_config)
    packets = [
        representative_frame(camera.camera_id, camera.source_path)
        for camera in load_camera_configs(project_root=PROJECT_ROOT)
        if camera.enabled
    ]

    for index in range(args.warmup):
        detector.detect(packets[index % len(packets)])
    latencies: list[float] = []
    counts: dict[str, list[int]] = {packet.camera_id: [] for packet in packets}
    for index in range(args.iterations):
        packet = packets[index % len(packets)]
        started = time.perf_counter()
        detections = detector.detect(packet)
        latencies.append((time.perf_counter() - started) * 1000.0)
        counts[packet.camera_id].append(len(detections))

    mean_ms = statistics.mean(latencies)
    print(
        f"backend={detector_config.backend} model={detector_config.model_path.name} "
        f"imgsz={detector_config.image_size} "
        f"threads={detector_config.cpu_threads} iterations={len(latencies)}"
    )
    print(
        f"mean_ms={mean_ms:.2f} median_ms={statistics.median(latencies):.2f} "
        f"p95_ms={percentile(latencies, 0.95):.2f} fps={1000.0 / mean_ms:.2f}"
    )
    for camera_id, person_counts in counts.items():
        print(f"{camera_id}: person_counts={person_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

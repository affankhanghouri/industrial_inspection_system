#!/usr/bin/env python3
"""Benchmark the PPE detector independently on representative CAM-001 frames."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.ppe import create_ppe_detector, load_ppe_config
from industrial_safety.video import FramePacket, load_camera_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--backend", choices=("ultralytics", "onnxruntime", "openvino"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations positive")
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
    positions = [
        round(index * (frame_count - 1) / 5) for index in range(6)
    ]
    packets: list[FramePacket] = []
    from datetime import datetime, timezone

    for frame_id, source_frame in enumerate(positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"could not decode frame {source_frame}")
        packets.append(
            FramePacket(
                camera_id="CAM-001",
                frame_id=frame_id,
                image=image,
                captured_at=datetime.now(timezone.utc),
                captured_monotonic=time.monotonic(),
                source_timestamp_ms=source_frame * 1000.0 / fps,
                source_frame_number=source_frame,
                loop_count=0,
                width=image.shape[1],
                height=image.shape[0],
            )
        )
    capture.release()

    detector_config = load_ppe_config(project_root=PROJECT_ROOT).detector
    updates = {}
    if args.backend is not None:
        updates["backend"] = args.backend
    if args.model is not None:
        updates["model_path"] = args.model.resolve()
    if args.imgsz is not None:
        updates["image_size"] = args.imgsz
    if args.confidence is not None:
        updates["confidence"] = args.confidence
    if args.threads is not None:
        updates["cpu_threads"] = args.threads
    detector_config = replace(detector_config, **updates)
    detector = create_ppe_detector(detector_config)
    detector.warmup()
    for index in range(args.warmup):
        detector.detect(packets[index % len(packets)])

    latencies: list[float] = []
    observation_counts: list[int] = []
    for index in range(args.iterations):
        packet = packets[index % len(packets)]
        started = time.perf_counter()
        observations = detector.detect(packet)
        latencies.append((time.perf_counter() - started) * 1000.0)
        observation_counts.append(len(observations))
    mean_ms = statistics.mean(latencies)
    print(f"backend={detector.config.backend} model={detector.config.model_path}")
    print(f"classes={sorted(detector.helmet_class_ids)} imgsz={detector.config.image_size}")
    print(
        f"iterations={len(latencies)} mean={mean_ms:.2f}ms "
        f"median={statistics.median(latencies):.2f}ms "
        f"p95={percentile(latencies, 0.95):.2f}ms "
        f"throughput={1000.0 / mean_ms:.2f} FPS"
    )
    print(
        f"helmet_observations min={min(observation_counts)} "
        f"max={max(observation_counts)} total={sum(observation_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

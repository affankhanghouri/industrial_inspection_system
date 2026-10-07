#!/usr/bin/env python3
"""Run live-style cameras with one shared person detector."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.inference import (
    RoundRobinInferenceScheduler,
    create_person_detector,
    load_inference_config,
)
from industrial_safety.inference.visualization import draw_inference_result
from industrial_safety.video import CameraManager, CameraState, load_camera_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cameras.yaml",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "inference.yaml",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stats-interval", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--cameras",
        nargs="+",
        metavar="CAMERA_ID",
        help="run only the listed camera IDs",
    )
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def log_health(
    camera_manager: CameraManager,
    scheduler: RoundRobinInferenceScheduler,
) -> None:
    camera_health = camera_manager.health()
    inference_health = scheduler.camera_stats()
    for camera_id, camera in camera_health.items():
        inference = inference_health[camera_id]
        logging.info(
            "[%s] state=%s camera_fps=%.1f camera_frame=%s ai_fps=%.2f "
            "processed=%s persons=%d mean_inference=%.1fms mean_age=%.1fms "
            "last_age=%s skipped=%d calls=%d errors=%d persons_total=%d persons_max=%d",
            camera_id,
            camera.state.value,
            camera.effective_reader_fps,
            camera.latest_frame_id,
            inference.inference_fps,
            inference.last_processed_frame_id,
            inference.last_person_count,
            inference.mean_inference_ms,
            inference.mean_end_to_end_age_ms,
            "n/a" if inference.last_end_to_end_age_ms is None else f"{inference.last_end_to_end_age_ms:.1f}ms",
            inference.frames_skipped,
            inference.inference_calls,
            inference.inference_errors,
            inference.total_person_detections,
            inference.max_person_count,
        )
    global_stats = scheduler.global_stats()
    logging.info(
        "[DETECTOR] throughput=%.2f FPS mean_inference=%.1fms calls=%d errors=%d",
        global_stats.inference_fps,
        global_stats.mean_inference_ms,
        global_stats.inference_calls,
        global_stats.inference_errors,
    )


def show_preview(scheduler: RoundRobinInferenceScheduler) -> bool:
    import cv2
    import numpy as np

    stats = scheduler.camera_stats()
    tiles = []
    for camera_id, result_buffer in scheduler.results.items():
        result = result_buffer.get_latest(timeout=0.0)
        packet = scheduler.latest_processed_frame(camera_id)
        if result is None or packet is None or result.frame_id != packet.frame_id:
            tile = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"{camera_id}: waiting for inference",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )
        else:
            annotated = draw_inference_result(
                packet.image,
                result,
                inference_fps=stats[camera_id].inference_fps,
            )
            tile = cv2.resize(annotated, (480, 270))
        tiles.append(tile)
    preview = np.hstack(tiles)
    cv2.imshow("Industrial Safety - Person Detection", preview)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main() -> int:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be greater than zero")
    if args.stats_interval <= 0:
        raise SystemExit("--stats-interval must be greater than zero")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    camera_configs = load_camera_configs(args.camera_config, project_root=PROJECT_ROOT)
    if args.cameras:
        requested = set(args.cameras)
        known = {camera.camera_id for camera in camera_configs}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"unknown camera IDs: {', '.join(sorted(unknown))}")
        camera_configs = [
            camera for camera in camera_configs if camera.camera_id in requested
        ]

    inference_config = load_inference_config(
        args.inference_config, project_root=PROJECT_ROOT
    )
    detector_config = inference_config.detector
    if args.confidence is not None:
        if not 0 < args.confidence <= 1:
            raise SystemExit("--confidence must be in (0, 1]")
        detector_config = replace(detector_config, confidence=args.confidence)
    if args.imgsz is not None:
        if args.imgsz <= 0:
            raise SystemExit("--imgsz must be greater than zero")
        detector_config = replace(detector_config, image_size=args.imgsz)
    if args.threads is not None:
        if args.threads <= 0:
            raise SystemExit("--threads must be greater than zero")
        detector_config = replace(detector_config, cpu_threads=args.threads)

    try:
        detector = create_person_detector(detector_config)
        warmup_started = time.monotonic()
        detector.warmup()
        logging.info(
            "detector warm-up completed in %.1fms: model=%s imgsz=%d threads=%d",
            (time.monotonic() - warmup_started) * 1000.0,
            detector_config.model_path.name,
            detector_config.image_size,
            detector_config.cpu_threads,
        )
    except Exception:
        logging.exception("detector initialization failed")
        return 2

    manager = CameraManager(camera_configs)
    scheduler = RoundRobinInferenceScheduler(
        manager.workers,
        detector,
        idle_wait_seconds=inference_config.runtime.idle_wait_seconds,
        retain_processed_frames=args.display,
    )
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    next_stats_at = started_at
    try:
        manager.start_all()
        scheduler.start()
        while not stop_requested.is_set():
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            if now >= next_stats_at:
                log_health(manager, scheduler)
                next_stats_at = now + args.stats_interval
            if args.display:
                if show_preview(scheduler):
                    break
                stop_requested.wait(0.01)
            else:
                stop_requested.wait(min(0.25, max(0.01, next_stats_at - now)))
    finally:
        scheduler.stop()
        scheduler.join(timeout=10.0)
        manager.stop_all()
        if args.display:
            import cv2

            cv2.destroyAllWindows()
        log_health(manager, scheduler)

    if scheduler.is_alive:
        logging.error("inference scheduler did not stop within 10 seconds")
        return 1
    if scheduler.fatal_error:
        return 1
    if any(health.state is CameraState.ERROR for health in manager.health().values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

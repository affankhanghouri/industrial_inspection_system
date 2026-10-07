#!/usr/bin/env python3
"""Run cameras, shared person detection, and camera-local person tracking."""

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
from industrial_safety.tracking import (
    TimeAwareIoUPersonTracker,
    TrackingCoordinator,
    load_tracking_config,
)
from industrial_safety.tracking.visualization import draw_tracking_result
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
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "tracking.yaml",
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
    inference_scheduler: RoundRobinInferenceScheduler,
    tracking_coordinator: TrackingCoordinator,
) -> None:
    camera_health = camera_manager.health()
    inference_health = inference_scheduler.camera_stats()
    tracking_health = tracking_coordinator.camera_stats()
    for camera_id, camera in camera_health.items():
        inference = inference_health[camera_id]
        tracking = tracking_health[camera_id]
        logging.info(
            "[%s] camera_fps=%.1f ai_fps=%.2f inference=%.1fms "
            "tracking_fps=%.2f tracks=%d confirmed=%d lost=%d created=%d "
            "expired=%d resets=%d "
            "tracking_mean=%.3fms tracking_p95=%.3fms obs_interval=%.3fs "
            "frame_gap_max=%d tracking_errors=%d",
            camera_id,
            camera.effective_reader_fps,
            inference.inference_fps,
            inference.mean_inference_ms,
            tracking.tracking_fps,
            tracking.current_active_tracks,
            tracking.current_confirmed_tracks,
            tracking.current_lost_tracks,
            tracking.total_tracks_created,
            tracking.expired_tracks,
            tracking.tracker_resets,
            tracking.mean_tracking_ms,
            tracking.p95_tracking_ms,
            tracking.mean_observation_interval_seconds,
            tracking.maximum_frame_gap,
            tracking.tracking_errors,
        )
    inference_global = inference_scheduler.global_stats()
    tracking_global = tracking_coordinator.global_stats()
    logging.info(
        "[PIPELINE] detector=%.2f FPS/%.1fms tracking=%.2f FPS/%d updates "
        "mean=%.3fms median=%.3fms p95=%.3fms max=%.3fms errors=%d",
        inference_global.inference_fps,
        inference_global.mean_inference_ms,
        tracking_global.tracking_fps,
        tracking_global.tracking_updates,
        tracking_global.mean_tracking_ms,
        tracking_global.median_tracking_ms,
        tracking_global.p95_tracking_ms,
        tracking_global.max_tracking_ms,
        tracking_global.tracking_errors,
    )


def show_preview(
    inference_scheduler: RoundRobinInferenceScheduler,
    tracking_coordinator: TrackingCoordinator,
) -> bool:
    import cv2
    import numpy as np

    inference_stats = inference_scheduler.camera_stats()
    tiles = []
    for camera_id, result_buffer in tracking_coordinator.results.items():
        result = result_buffer.get_latest(timeout=0.0)
        packet = inference_scheduler.latest_processed_frame(camera_id)
        if result is None or packet is None or result.frame_id != packet.frame_id:
            tile = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"{camera_id}: waiting for tracks",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )
        else:
            annotated = draw_tracking_result(
                packet.image,
                result,
                inference_fps=inference_stats[camera_id].inference_fps,
            )
            tile = cv2.resize(annotated, (480, 270))
        tiles.append(tile)
    preview = np.hstack(tiles)
    cv2.imshow("Industrial Safety - Person Tracking", preview)
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
    tracking_config = load_tracking_config(
        args.tracking_config, project_root=PROJECT_ROOT
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

    camera_manager = CameraManager(camera_configs)
    inference_scheduler = RoundRobinInferenceScheduler(
        camera_manager.workers,
        detector,
        idle_wait_seconds=inference_config.runtime.idle_wait_seconds,
        retain_processed_frames=args.display,
    )
    tracking_coordinator = TrackingCoordinator(
        inference_scheduler.results,
        tracker_factory=lambda camera_id: TimeAwareIoUPersonTracker(
            camera_id, tracking_config.tracker
        ),
        idle_wait_seconds=tracking_config.runtime.idle_wait_seconds,
    )
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    next_stats_at = started_at
    try:
        camera_manager.start_all()
        inference_scheduler.start()
        tracking_coordinator.start()
        while not stop_requested.is_set():
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            if now >= next_stats_at:
                log_health(
                    camera_manager, inference_scheduler, tracking_coordinator
                )
                next_stats_at = now + args.stats_interval
            if args.display:
                if show_preview(inference_scheduler, tracking_coordinator):
                    break
                stop_requested.wait(0.01)
            else:
                stop_requested.wait(min(0.25, max(0.01, next_stats_at - now)))
    finally:
        inference_scheduler.stop()
        inference_scheduler.join(timeout=10.0)
        tracking_coordinator.stop()
        tracking_coordinator.join(timeout=5.0)
        camera_manager.stop_all()
        if args.display:
            import cv2

            cv2.destroyAllWindows()
        log_health(camera_manager, inference_scheduler, tracking_coordinator)

    if inference_scheduler.is_alive or tracking_coordinator.is_alive:
        logging.error("pipeline component did not stop within its timeout")
        return 1
    if inference_scheduler.fatal_error or tracking_coordinator.fatal_error:
        return 1
    if any(
        health.state is CameraState.ERROR
        for health in camera_manager.health().values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

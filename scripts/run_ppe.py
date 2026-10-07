#!/usr/bin/env python3
"""Run person tracking plus camera-selective helmet perception."""

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
from industrial_safety.ppe import (
    HelmetTrackAssociator,
    PPECoordinator,
    PPEObservationState,
    create_ppe_detector,
    load_ppe_config,
)
from industrial_safety.ppe.visualization import draw_ppe_association
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
    parser.add_argument(
        "--ppe-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ppe.yaml",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stats-interval", type=float, default=3.0)
    parser.add_argument("--ppe-confidence", type=float, default=None)
    parser.add_argument("--ppe-imgsz", type=int, default=None)
    parser.add_argument("--ppe-min-interval", type=float, default=None)
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
    ppe_coordinator: PPECoordinator,
) -> None:
    cameras = camera_manager.health()
    inference = inference_scheduler.camera_stats()
    tracking = tracking_coordinator.camera_stats()
    ppe = ppe_coordinator.camera_stats()
    for camera_id, camera in cameras.items():
        base = inference[camera_id]
        track = tracking[camera_id]
        if camera_id not in ppe:
            logging.info(
                "[%s] camera_fps=%.1f person_fps=%.2f person=%.1fms "
                "tracking_fps=%.2f tracking=%.3fms PPE=disabled",
                camera_id,
                camera.effective_reader_fps,
                base.inference_fps,
                base.mean_inference_ms,
                track.tracking_fps,
                track.mean_tracking_ms,
            )
            continue
        helmet = ppe[camera_id]
        latest = ppe_coordinator.association_results[camera_id].get_latest(
            timeout=0.0
        )
        latest_observed = (
            0
            if latest is None
            else sum(
                item.state is PPEObservationState.OBSERVED
                for item in latest.track_observations
            )
        )
        latest_unobserved = (
            0
            if latest is None
            else sum(
                item.state is PPEObservationState.UNOBSERVED
                for item in latest.track_observations
            )
        )
        latest_raw = (
            0
            if latest is None
            else len(latest.ppe_inference_result.observations)
        )
        logging.info(
            "[%s] camera_fps=%.1f person_fps=%.2f person=%.1fms "
            "tracking_fps=%.2f tracking=%.3fms PPE_fps=%.2f PPE=%.1fms "
            "association=%.3fms PPE_age=%.1fms raw_helmets=%d "
            "observed=%d unobserved=%d mismatches=%d PPE_errors=%d",
            camera_id,
            camera.effective_reader_fps,
            base.inference_fps,
            base.mean_inference_ms,
            track.tracking_fps,
            track.mean_tracking_ms,
            helmet.ppe_fps,
            helmet.mean_inference_ms,
            helmet.mean_association_ms,
            helmet.mean_end_to_end_age_ms,
            latest_raw,
            latest_observed,
            latest_unobserved,
            helmet.frame_mismatches,
            helmet.inference_errors,
        )
    person_global = inference_scheduler.global_stats()
    tracking_global = tracking_coordinator.global_stats()
    ppe_global = ppe_coordinator.global_stats()
    logging.info(
        "[PIPELINE] person=%.2f FPS/%.1fms tracking=%.2f FPS/%.3fms "
        "PPE=%.2f FPS mean=%.1fms median=%.1fms p95=%.1fms "
        "association mean=%.3fms median=%.3fms p95=%.3fms PPE_age=%.1fms "
        "PPE_errors=%d",
        person_global.inference_fps,
        person_global.mean_inference_ms,
        tracking_global.tracking_fps,
        tracking_global.mean_tracking_ms,
        ppe_global.ppe_fps,
        ppe_global.mean_inference_ms,
        ppe_global.median_inference_ms,
        ppe_global.p95_inference_ms,
        ppe_global.mean_association_ms,
        ppe_global.median_association_ms,
        ppe_global.p95_association_ms,
        ppe_global.mean_end_to_end_age_ms,
        ppe_global.inference_errors,
    )


def show_preview(
    inference_scheduler: RoundRobinInferenceScheduler,
    tracking_coordinator: TrackingCoordinator,
    ppe_coordinator: PPECoordinator,
) -> bool:
    import cv2
    import numpy as np

    tiles = []
    enabled = set(ppe_coordinator.enabled_camera_ids)
    for camera_id, tracking_buffer in tracking_coordinator.results.items():
        if camera_id in enabled:
            result = ppe_coordinator.association_results[camera_id].get_latest(
                timeout=0.0
            )
            packet = ppe_coordinator.latest_processed_frame(camera_id)
            valid = (
                result is not None
                and packet is not None
                and result.frame_id == packet.frame_id
            )
            annotated = (
                draw_ppe_association(packet.image, result) if valid else None
            )
        else:
            result = tracking_buffer.get_latest(timeout=0.0)
            packet = inference_scheduler.latest_processed_frame(camera_id)
            valid = (
                result is not None
                and packet is not None
                and result.frame_id == packet.frame_id
            )
            annotated = (
                draw_tracking_result(packet.image, result) if valid else None
            )
        if annotated is None:
            tile = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"{camera_id}: waiting for exact result",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 220, 255),
                2,
            )
        else:
            tile = cv2.resize(annotated, (480, 270))
        tiles.append(tile)
    cv2.imshow("Industrial Safety - Helmet Observations", np.hstack(tiles))
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
    selected_ids = {camera.camera_id for camera in camera_configs if camera.enabled}

    inference_config = load_inference_config(
        args.inference_config, project_root=PROJECT_ROOT
    )
    tracking_config = load_tracking_config(
        args.tracking_config, project_root=PROJECT_ROOT
    )
    ppe_config = load_ppe_config(args.ppe_config, project_root=PROJECT_ROOT)
    enabled_ppe_cameras = tuple(
        camera_id
        for camera_id in ppe_config.runtime.enabled_camera_ids
        if camera_id in selected_ids
    )
    if not enabled_ppe_cameras:
        raise SystemExit("none of the selected cameras has PPE enabled")

    ppe_detector_config = ppe_config.detector
    if args.ppe_confidence is not None:
        if not 0 < args.ppe_confidence <= 1:
            raise SystemExit("--ppe-confidence must be in (0, 1]")
        ppe_detector_config = replace(
            ppe_detector_config, confidence=args.ppe_confidence
        )
    if args.ppe_imgsz is not None:
        if args.ppe_imgsz <= 0:
            raise SystemExit("--ppe-imgsz must be greater than zero")
        ppe_detector_config = replace(
            ppe_detector_config, image_size=args.ppe_imgsz
        )
    minimum_interval = ppe_config.runtime.minimum_interval_seconds
    if args.ppe_min_interval is not None:
        if args.ppe_min_interval < 0:
            raise SystemExit("--ppe-min-interval cannot be negative")
        minimum_interval = args.ppe_min_interval

    try:
        person_detector = create_person_detector(inference_config.detector)
        started = time.monotonic()
        person_detector.warmup()
        logging.info(
            "person detector warm-up completed in %.1fms: model=%s",
            (time.monotonic() - started) * 1000.0,
            inference_config.detector.model_path.name,
        )
        ppe_detector = create_ppe_detector(ppe_detector_config)
        started = time.monotonic()
        ppe_detector.warmup()
        logging.info(
            "PPE detector warm-up completed in %.1fms: model=%s classes=%s",
            (time.monotonic() - started) * 1000.0,
            ppe_detector_config.model_path.name,
            sorted(ppe_detector.helmet_class_ids),
        )
    except Exception:
        logging.exception("detector initialization failed")
        return 2

    camera_manager = CameraManager(camera_configs)
    inference_scheduler = RoundRobinInferenceScheduler(
        camera_manager.workers,
        person_detector,
        idle_wait_seconds=inference_config.runtime.idle_wait_seconds,
        retain_processed_frames=True,
    )
    tracking_coordinator = TrackingCoordinator(
        inference_scheduler.results,
        tracker_factory=lambda camera_id: TimeAwareIoUPersonTracker(
            camera_id, tracking_config.tracker
        ),
        idle_wait_seconds=tracking_config.runtime.idle_wait_seconds,
    )
    ppe_coordinator = PPECoordinator(
        tracking_coordinator.results,
        inference_scheduler.latest_processed_frame,
        ppe_detector,
        HelmetTrackAssociator(ppe_config.association),
        enabled_camera_ids=enabled_ppe_cameras,
        minimum_interval_seconds=minimum_interval,
        idle_wait_seconds=ppe_config.runtime.idle_wait_seconds,
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
        camera_manager.start_all()
        inference_scheduler.start()
        tracking_coordinator.start()
        ppe_coordinator.start()
        while not stop_requested.is_set():
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            if now >= next_stats_at:
                log_health(
                    camera_manager,
                    inference_scheduler,
                    tracking_coordinator,
                    ppe_coordinator,
                )
                next_stats_at = now + args.stats_interval
            if args.display:
                if show_preview(
                    inference_scheduler, tracking_coordinator, ppe_coordinator
                ):
                    break
                stop_requested.wait(0.01)
            else:
                stop_requested.wait(min(0.25, max(0.01, next_stats_at - now)))
    finally:
        inference_scheduler.stop()
        inference_scheduler.join(timeout=10.0)
        ppe_coordinator.stop()
        ppe_coordinator.join(timeout=10.0)
        tracking_coordinator.stop()
        tracking_coordinator.join(timeout=5.0)
        camera_manager.stop_all()
        if args.display:
            import cv2

            cv2.destroyAllWindows()
        log_health(
            camera_manager,
            inference_scheduler,
            tracking_coordinator,
            ppe_coordinator,
        )

    if any(
        component.is_alive
        for component in (
            inference_scheduler,
            tracking_coordinator,
            ppe_coordinator,
        )
    ):
        logging.error("pipeline component did not stop within its timeout")
        return 1
    if any(
        error is not None
        for error in (
            inference_scheduler.fatal_error,
            tracking_coordinator.fatal_error,
            ppe_coordinator.fatal_error,
        )
    ):
        return 1
    if any(
        health.state is CameraState.ERROR
        for health in camera_manager.health().values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

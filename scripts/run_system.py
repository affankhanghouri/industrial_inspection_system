#!/usr/bin/env python3
"""Launch the complete safety demonstration and dashboard."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.api import create_app
from industrial_safety.events import EvidenceConfig, EvidenceRecorder
from industrial_safety.events.service import EventService
from industrial_safety.inference import (
    RoundRobinInferenceScheduler,
    create_person_detector,
    load_inference_config,
)
from industrial_safety.ppe import (
    HelmetTrackAssociator,
    PPECoordinator,
    create_ppe_detector,
    load_ppe_config,
)
from industrial_safety.rules import RuleEngine, load_rules
from industrial_safety.runtime import SafetyCoordinator, draw_safety_preview
from industrial_safety.storage import SQLiteEventStore
from industrial_safety.tracking import (
    TimeAwareIoUPersonTracker,
    TrackingCoordinator,
    load_tracking_config,
)
from industrial_safety.video import (
    CameraConfig,
    CameraManager,
    FramePacket,
    load_camera_configs,
)
from industrial_safety.zones import Zone, ZoneEngine, load_zones


DISPLAY_WINDOW = "Industrial Safety - Live Camera"
MULTI_DISPLAY_WINDOW = "Industrial Safety - Multi Camera"
MULTI_DISPLAY_FPS = 15.0
MULTI_TILE_WIDTH = 640
MULTI_TILE_HEIGHT = 360
MULTI_TILE_HEADER_HEIGHT = 28


def display_window_name(camera_id: str) -> str:
    return f"{DISPLAY_WINDOW} [{camera_id}]"


def display_mode(args: argparse.Namespace) -> str:
    if not args.display:
        return "headless"
    return "multi" if args.camera is None else "single"


class DisplayTelemetry:
    """Main-thread counters for rendered and visibly advancing frames."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.started_monotonic = time.monotonic()
        self.rendered_frames = 0
        self.source_frame_updates = 0
        self.last_frame_id: int | None = None
        self.screenshot_saved = False

    def record(self, frame_id: int | None) -> None:
        self.rendered_frames += 1
        if frame_id is not None and frame_id != self.last_frame_id:
            self.source_frame_updates += 1
            self.last_frame_id = frame_id

    def rates(self) -> tuple[float, float]:
        elapsed = max(1e-9, time.monotonic() - self.started_monotonic)
        return self.rendered_frames / elapsed, self.source_frame_updates / elapsed

    def log(self) -> None:
        render_fps, video_fps = self.rates()
        if self.camera_id == "MULTI":
            logging.info("[DISPLAY] multi_camera_refresh=%.2fFPS", render_fps)
            return
        logging.info(
            "[%s] display=%.2fFPS advancing_video=%.2fFPS latest_frame=%s",
            self.camera_id,
            render_fps,
            video_fps,
            self.last_frame_id,
        )


class MultiCameraDisplayState:
    """Cache tiles and update two independent cameras per display refresh."""

    def __init__(self) -> None:
        self.canvases: dict[str, object] = {}
        self.next_camera_index = 0

    def cameras_to_refresh(self, camera_ids: tuple[str, ...]) -> tuple[str, ...]:
        missing = tuple(camera_id for camera_id in camera_ids if camera_id not in self.canvases)
        if missing:
            return missing
        count = min(2, len(camera_ids))
        selected = tuple(
            camera_ids[(self.next_camera_index + offset) % len(camera_ids)]
            for offset in range(count)
        )
        self.next_camera_index = (self.next_camera_index + count) % len(camera_ids)
        return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stats-interval", type=float, default=5.0)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/safety.db")
    parser.add_argument(
        "--evidence-dir", type=Path, default=PROJECT_ROOT / "data/evidence"
    )
    parser.add_argument("--evidence-fps", type=float, default=10.0)
    parser.add_argument("--evidence-pre-seconds", type=float, default=2.5)
    parser.add_argument("--evidence-post-seconds", type=float, default=2.5)
    parser.add_argument("--evidence-max-width", type=int, default=640)
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--display-screenshot",
        type=Path,
        help="save the first event-bearing rendered display frame",
    )
    parser.add_argument(
        "--camera",
        metavar="CAMERA_ID",
        help="run only one configured camera, for example CAM-001",
    )
    return parser.parse_args(argv)


def select_camera_configs(
    configs: Sequence[CameraConfig], camera_id: str | None
) -> tuple[CameraConfig, ...]:
    """Validate and apply the optional single-camera runtime selection."""

    if camera_id is None:
        return tuple(configs)
    known = {config.camera_id: config for config in configs}
    if camera_id not in known:
        choices = ", ".join(sorted(known))
        raise SystemExit(
            f"unknown camera ID {camera_id!r}; configured cameras: {choices}"
        )
    selected = known[camera_id]
    if not selected.enabled:
        raise SystemExit(f"camera {camera_id!r} is disabled in configuration")
    return (selected,)


def main() -> int:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.display_screenshot is not None and not args.display:
        raise SystemExit("--display-screenshot requires --display")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.display:
        # Preview resizing/drawing must not fan out across every CPU core and
        # compete with the ONNX/OpenVINO inference workers.
        import cv2

        cv2.setNumThreads(1)

    camera_configs = select_camera_configs(
        load_camera_configs(project_root=PROJECT_ROOT), args.camera
    )
    selected_ids = {config.camera_id for config in camera_configs if config.enabled}
    if not selected_ids:
        raise SystemExit("no enabled cameras selected")
    inference_config = load_inference_config(project_root=PROJECT_ROOT)
    tracking_config = load_tracking_config(project_root=PROJECT_ROOT)
    ppe_config = load_ppe_config(project_root=PROJECT_ROOT)
    person_detector = create_person_detector(inference_config.detector)
    started = time.perf_counter()
    person_detector.warmup()
    logging.info(
        "person backend ready in %.1fms: %s %dpx",
        (time.perf_counter() - started) * 1000.0,
        inference_config.detector.backend,
        inference_config.detector.image_size,
    )

    camera_manager = CameraManager(camera_configs)
    camera_workers = camera_manager.workers
    evidence_config = EvidenceConfig(
        fps=args.evidence_fps,
        pre_seconds=args.evidence_pre_seconds,
        post_seconds=args.evidence_post_seconds,
        max_width=args.evidence_max_width,
    )

    def latest_camera_frame(camera_id: str):
        return camera_workers[camera_id].latest_frame.get_latest(timeout=0.0)

    evidence_recorder = EvidenceRecorder(
        selected_ids,
        latest_camera_frame,
        args.evidence_dir,
        config=evidence_config,
    )
    inference = RoundRobinInferenceScheduler(
        camera_workers,
        person_detector,
        idle_wait_seconds=inference_config.runtime.idle_wait_seconds,
        retain_processed_frames=True,
    )
    tracking = TrackingCoordinator(
        inference.results,
        tracker_factory=lambda camera_id: TimeAwareIoUPersonTracker(
            camera_id, tracking_config.tracker
        ),
        idle_wait_seconds=tracking_config.runtime.idle_wait_seconds,
    )
    enabled_ppe_cameras = tuple(
        camera_id
        for camera_id in ppe_config.runtime.enabled_camera_ids
        if camera_id in selected_ids
    )
    ppe: PPECoordinator | None = None
    if enabled_ppe_cameras:
        ppe_detector = create_ppe_detector(ppe_config.detector)
        started = time.perf_counter()
        ppe_detector.warmup()
        logging.info(
            "PPE backend ready in %.1fms: %s %dpx",
            (time.perf_counter() - started) * 1000.0,
            ppe_config.detector.backend,
            ppe_config.detector.image_size,
        )
        ppe = PPECoordinator(
            tracking.results,
            inference.latest_processed_frame,
            ppe_detector,
            HelmetTrackAssociator(ppe_config.association),
            enabled_camera_ids=enabled_ppe_cameras,
            minimum_interval_seconds=ppe_config.runtime.minimum_interval_seconds,
            idle_wait_seconds=ppe_config.runtime.idle_wait_seconds,
            retain_processed_frames=True,
        )
    store = SQLiteEventStore(args.database)
    events = EventService(
        store,
        args.evidence_dir,
        person_model_version=(
            f"{inference_config.detector.model_path.name}:"
            f"{inference_config.detector.backend}:{inference_config.detector.image_size}"
        ),
        ppe_model_version=(
            f"{ppe_config.detector.model_path.name}:"
            f"{ppe_config.detector.backend}:{ppe_config.detector.image_size}"
        ),
        evidence_recorder=evidence_recorder,
    )
    rules = RuleEngine(load_rules(project_root=PROJECT_ROOT))
    zones = tuple(
        zone
        for zone in load_zones(project_root=PROJECT_ROOT)
        if zone.camera_id in selected_ids
    )
    safety = SafetyCoordinator(
        tracking.results,
        {} if ppe is None else ppe.association_results,
        ZoneEngine(zones),
        rules,
        events,
        inference.latest_processed_frame,
        inference.latest_processed_frame if ppe is None else ppe.latest_processed_frame,
    )

    names = {config.camera_id: config.name for config in camera_configs}
    locations = {
        "CAM-001": "Refinery Area",
        "CAM-002": "Loading Area",
        "CAM-003": "Restricted Area",
    }

    def camera_health() -> dict[str, dict[str, object]]:
        camera_stats = camera_manager.health()
        inference_stats = inference.camera_stats()
        tracking_stats = tracking.camera_stats()
        ppe_stats = {} if ppe is None else ppe.camera_stats()
        return {
            camera_id: {
                "camera_id": camera_id,
                "name": names[camera_id],
                "location": locations.get(camera_id, names[camera_id]),
                "state": camera.state.value,
                "camera_fps": camera.effective_reader_fps,
                "frame_age_ms": (
                    None
                    if camera.latest_frame_age_seconds is None
                    else camera.latest_frame_age_seconds * 1000.0
                ),
                "person_fps": inference_stats[camera_id].inference_fps,
                "person_latency_ms": inference_stats[camera_id].mean_inference_ms,
                "tracking_fps": tracking_stats[camera_id].tracking_fps,
                "tracking_latency_ms": tracking_stats[camera_id].mean_tracking_ms,
                "ppe_fps": (
                    ppe_stats[camera_id].ppe_fps if camera_id in ppe_stats else None
                ),
            }
            for camera_id, camera in camera_stats.items()
        }

    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    if not args.no_api:
        app = create_app(
            store,
            args.evidence_dir,
            camera_health_provider=camera_health,
            web_root=PROJECT_ROOT / "web",
        )
        server = uvicorn.Server(
            uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
        )
        server_thread = threading.Thread(
            target=server.run, name="dashboard-api", daemon=False
        )

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    next_stats = started_at + args.stats_interval
    active_display_mode = display_mode(args)
    display_camera_id = args.camera or next(
        config.camera_id for config in camera_configs if config.enabled
    )
    display_telemetry = (
        DisplayTelemetry(
            "MULTI" if active_display_mode == "multi" else display_camera_id
        )
        if args.display
        else None
    )
    next_multi_display = started_at
    multi_display_state = MultiCameraDisplayState()
    exit_code = 0
    try:
        camera_manager.start_all()
        evidence_recorder.start()
        inference.start()
        tracking.start()
        if ppe is not None:
            ppe.start()
        safety.start()
        if server_thread is not None:
            server_thread.start()
            startup_deadline = time.monotonic() + 5.0
            while (
                server is not None
                and not server.started
                and server_thread.is_alive()
                and time.monotonic() < startup_deadline
            ):
                time.sleep(0.02)
            if server is None or not server.started:
                raise RuntimeError(
                    f"dashboard API failed to start on {args.host}:{args.port}"
                )
            logging.info("dashboard: http://%s:%d", args.host, args.port)
        if active_display_mode == "multi":
            try:
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
                logging.info("multi-camera preview thread set to low CPU priority")
            except (AttributeError, OSError):
                logging.debug("could not lower preview thread priority", exc_info=True)
        while not stop_requested.wait(0.01 if args.display else 0.1):
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            display_requested_shutdown = False
            if active_display_mode == "single":
                display_requested_shutdown = _show_preview(
                    display_camera_id,
                    zones,
                    camera_manager,
                    inference,
                    tracking,
                    ppe,
                    safety,
                    display_telemetry,
                    args.display_screenshot,
                )
            elif active_display_mode == "multi" and now >= next_multi_display:
                display_requested_shutdown = _show_multi_preview(
                    tuple(config.camera_id for config in camera_configs if config.enabled),
                    locations,
                    zones,
                    camera_manager,
                    inference,
                    tracking,
                    ppe,
                    safety,
                    multi_display_state,
                    display_telemetry,
                    args.display_screenshot,
                )
                next_multi_display = now + 1.0 / MULTI_DISPLAY_FPS
            if display_requested_shutdown:
                logging.info("display requested shutdown")
                stop_requested.set()
                break
            if now >= next_stats:
                _log_stats(camera_health(), inference, ppe, safety, store)
                if display_telemetry is not None:
                    display_telemetry.log()
                next_stats = now + args.stats_interval
    finally:
        safety.stop()
        safety.join(5.0)
        evidence_recorder.stop(flush=True)
        if ppe is not None:
            ppe.stop()
            ppe.join(10.0)
        tracking.stop()
        tracking.join(5.0)
        inference.stop()
        inference.join(10.0)
        camera_manager.stop_all()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(5.0)
        if args.display:
            try:
                import cv2

                cv2.destroyWindow(
                    MULTI_DISPLAY_WINDOW
                    if active_display_mode == "multi"
                    else display_window_name(display_camera_id)
                )
            except Exception:
                logging.debug("display window was already closed", exc_info=True)
        _log_stats(camera_health(), inference, ppe, safety, store)
        if display_telemetry is not None:
            display_telemetry.log()
        components = (safety, tracking, inference, evidence_recorder) + (() if ppe is None else (ppe,))
        if any(component.is_alive for component in components):
            logging.error("one or more pipeline components failed to stop")
            exit_code = 1
        if any(
            error is not None
            for error in (
                safety.fatal_error,
                None if ppe is None else ppe.fatal_error,
                tracking.fatal_error,
                inference.fatal_error,
            )
        ):
            exit_code = 1
        store.close()
    return exit_code


def _log_stats(
    cameras: dict[str, dict[str, object]],
    inference: RoundRobinInferenceScheduler,
    ppe: PPECoordinator | None,
    safety: SafetyCoordinator,
    store: SQLiteEventStore,
) -> None:
    person_global = inference.global_stats()
    ppe_global = None if ppe is None else ppe.global_stats()
    safety_stats = safety.stats()
    logging.info(
        "[SYSTEM] camera_fps=%s person=%.2fFPS/%.1fms PPE=%.2fFPS/%.1fms "
        "rules=%.3fms events=%d open=%d",
        "/".join(f"{float(item['camera_fps']):.1f}" for item in cameras.values()),
        person_global.inference_fps,
        person_global.mean_inference_ms,
        0.0 if ppe_global is None else ppe_global.ppe_fps,
        0.0 if ppe_global is None else ppe_global.mean_inference_ms,
        float(safety_stats["mean_processing_ms"]),
        store.stats().get("total", 0),
        store.stats().get("open", 0),
    )


def _show_preview(
    camera_id: str,
    zones: tuple[Zone, ...],
    camera_manager: CameraManager,
    inference: RoundRobinInferenceScheduler,
    tracking: TrackingCoordinator,
    ppe: PPECoordinator | None,
    safety: SafetyCoordinator,
    display_telemetry: DisplayTelemetry | None = None,
    display_screenshot: Path | None = None,
) -> bool:
    """Show one camera using already-published final-pipeline results."""

    import cv2

    worker = camera_manager.workers[camera_id]
    packet = worker.latest_frame.get_latest(timeout=0.0)
    window_name = display_window_name(camera_id)
    if (
        packet is not None
        and display_telemetry is not None
        and display_telemetry.last_frame_id == packet.frame_id
    ):
        return _display_requests_shutdown(cv2, window_name)

    canvas, latest_event = _render_camera_preview(
        camera_id,
        packet,
        zones,
        camera_manager,
        inference,
        tracking,
        ppe,
        safety,
    )
    if (
        display_screenshot is not None
        and latest_event is not None
        and display_telemetry is not None
        and not display_telemetry.screenshot_saved
    ):
        display_screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(display_screenshot), canvas):
            raise RuntimeError(f"failed to save display screenshot: {display_screenshot}")
        display_telemetry.screenshot_saved = True
        logging.info("[%s] display screenshot: %s", camera_id, display_screenshot)

    cv2.imshow(window_name, canvas)
    if display_telemetry is not None:
        display_telemetry.record(None if packet is None else packet.frame_id)
    return _display_requests_shutdown(cv2, window_name)


def _render_camera_preview(
    camera_id: str,
    packet: FramePacket | None,
    zones: tuple[Zone, ...],
    camera_manager: CameraManager,
    inference: RoundRobinInferenceScheduler,
    tracking: TrackingCoordinator,
    ppe: PPECoordinator | None,
    safety: SafetyCoordinator,
):
    """Render one latest camera frame using the established visualization."""

    import cv2
    import numpy as np

    # Camera imagery and every overlay result are read independently. No
    # display path waits for matching frame IDs or for PPE inference.
    tracking_result = tracking.results[camera_id].get_latest(timeout=0.0)
    if (
        packet is not None
        and tracking_result is not None
        and tracking_result.inference_result.source_loop_count != packet.loop_count
    ):
        tracking_result = None

    ppe_result = None
    if ppe is not None and camera_id in ppe.enabled_camera_ids:
        ppe_result = ppe.association_results[camera_id].get_latest(timeout=0.0)
        if (
            packet is not None
            and ppe_result is not None
            and ppe_result.ppe_inference_result.source_loop_count != packet.loop_count
        ):
            ppe_result = None

    if packet is None:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            f"{camera_id}: waiting for synchronized pipeline result",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 220, 255),
            2,
        )
        latest_event = None
    else:
        camera_stats = camera_manager.health()[camera_id]
        person_stats = inference.camera_stats()[camera_id]
        ppe_stats = None if ppe is None else ppe.camera_stats().get(camera_id)
        latest_event = safety.latest_event(camera_id)
        now = time.monotonic()
        canvas = draw_safety_preview(
            packet.image,
            tracking_result,
            zones=tuple(zone for zone in zones if zone.camera_id == camera_id),
            zone_evaluation=safety.latest_zone_evaluation(camera_id),
            ppe_result=ppe_result,
            latest_event=latest_event,
            camera_id=camera_id,
            camera_fps=camera_stats.effective_reader_fps,
            person_fps=person_stats.inference_fps,
            ppe_fps=None if ppe_stats is None else ppe_stats.ppe_fps,
            person_age_ms=(
                None
                if tracking_result is None
                else max(
                    0.0,
                    (
                        now
                        - tracking_result.inference_result.inference_finished_monotonic
                    )
                    * 1000.0,
                )
            ),
            ppe_age_ms=(
                None
                if ppe_result is None
                else max(
                    0.0,
                    (now - ppe_result.ppe_inference_result.inference_finished_monotonic)
                    * 1000.0,
                )
            ),
        )
    return canvas, latest_event


def compose_multi_camera_preview(
    frames: dict[str, object | None],
    camera_labels: dict[str, str],
):
    """Compose three rendered previews without any non-video tile."""

    import cv2
    import numpy as np

    camera_ids = tuple(frames)
    if len(camera_ids) > 3:
        raise ValueError("multi-camera preview supports at most three camera tiles")
    canvas = np.full(
        (MULTI_TILE_HEIGHT * 2, MULTI_TILE_WIDTH * 2, 3),
        (13, 18, 24),
        dtype=np.uint8,
    )
    positions = (
        (0, 0),
        (MULTI_TILE_WIDTH, 0),
        (MULTI_TILE_WIDTH // 2, MULTI_TILE_HEIGHT),
    )
    for camera_id, (left, top) in zip(camera_ids, positions, strict=False):
        tile = canvas[
            top : top + MULTI_TILE_HEIGHT,
            left : left + MULTI_TILE_WIDTH,
        ]
        _draw_camera_tile(tile, frames[camera_id], camera_id, camera_labels[camera_id])

    cv2.line(
        canvas,
        (MULTI_TILE_WIDTH, 0),
        (MULTI_TILE_WIDTH, MULTI_TILE_HEIGHT),
        (62, 70, 78),
        1,
    )
    cv2.line(canvas, (0, MULTI_TILE_HEIGHT), (1280, MULTI_TILE_HEIGHT), (62, 70, 78), 1)
    return canvas


def _draw_camera_tile(tile, frame: object | None, camera_id: str, label: str) -> None:
    import cv2

    tile[:] = (16, 22, 29)
    cv2.rectangle(
        tile,
        (0, 0),
        (MULTI_TILE_WIDTH, MULTI_TILE_HEADER_HEIGHT),
        (33, 48, 62),
        -1,
    )
    cv2.putText(
        tile,
        f"{camera_id} | {label}",
        (11, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if frame is None:
        cv2.putText(
            tile,
            "Waiting for camera...",
            (190, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return

    image = frame
    source_height, source_width = image.shape[:2]
    available_height = MULTI_TILE_HEIGHT - MULTI_TILE_HEADER_HEIGHT
    scale = min(MULTI_TILE_WIDTH / source_width, available_height / source_height)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    # Nearest-neighbor scaling keeps this non-recording preview lightweight;
    # evidence and the underlying camera frames retain their original quality.
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    left = (MULTI_TILE_WIDTH - width) // 2
    top = MULTI_TILE_HEADER_HEIGHT + (available_height - height) // 2
    tile[top : top + height, left : left + width] = resized


def _show_multi_preview(
    camera_ids: tuple[str, ...],
    camera_labels: dict[str, str],
    zones: tuple[Zone, ...],
    camera_manager: CameraManager,
    inference: RoundRobinInferenceScheduler,
    tracking: TrackingCoordinator,
    ppe: PPECoordinator | None,
    safety: SafetyCoordinator,
    display_state: MultiCameraDisplayState,
    display_telemetry: DisplayTelemetry | None = None,
    display_screenshot: Path | None = None,
) -> bool:
    import cv2

    workers = camera_manager.workers
    for camera_id in display_state.cameras_to_refresh(camera_ids):
        packet = workers[camera_id].latest_frame.get_latest(timeout=0.0)
        canvas, latest_event = _render_camera_preview(
            camera_id,
            packet,
            zones,
            camera_manager,
            inference,
            tracking,
            ppe,
            safety,
        )
        display_state.canvases[camera_id] = canvas

    has_event = any(safety.latest_event(camera_id) is not None for camera_id in camera_ids)

    composite = compose_multi_camera_preview(
        {camera_id: display_state.canvases.get(camera_id) for camera_id in camera_ids},
        camera_labels,
    )
    if (
        display_screenshot is not None
        and has_event
        and display_telemetry is not None
        and not display_telemetry.screenshot_saved
    ):
        display_screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(display_screenshot), composite):
            raise RuntimeError(f"failed to save display screenshot: {display_screenshot}")
        display_telemetry.screenshot_saved = True
        logging.info("multi-camera display screenshot: %s", display_screenshot)

    cv2.imshow(MULTI_DISPLAY_WINDOW, composite)
    if display_telemetry is not None:
        display_telemetry.record(display_telemetry.rendered_frames)
    return _display_requests_shutdown(cv2, MULTI_DISPLAY_WINDOW)


def _display_requests_shutdown(
    cv2_module: object, window_name: str = DISPLAY_WINDOW
) -> bool:
    key = cv2_module.waitKey(1) & 0xFF  # type: ignore[attr-defined]
    if key == ord("q"):
        return True
    try:
        return (
            cv2_module.getWindowProperty(  # type: ignore[attr-defined]
                window_name, cv2_module.WND_PROP_VISIBLE  # type: ignore[attr-defined]
            )
            < 1
        )
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(main())

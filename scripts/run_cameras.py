#!/usr/bin/env python3
"""Run configured cameras as continuous, live-style streams."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.video import CameraManager, CameraState, load_camera_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cameras.yaml",
        help="camera YAML path",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="show a tiled preview; headless operation is the default",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop automatically after this many seconds (useful for smoke tests)",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=3.0,
        help="seconds between health log lines",
    )
    return parser.parse_args()


def log_health(manager: CameraManager) -> None:
    for camera_id, health in manager.health().items():
        age = "n/a" if health.latest_frame_age_seconds is None else f"{health.latest_frame_age_seconds:.3f}s"
        logging.info(
            "[%s] state=%s frames=%d latest=%s age=%s reader_fps=%.1f source_fps=%.1f loops=%d%s",
            camera_id,
            health.state.value,
            health.frames_read,
            health.latest_frame_id,
            age,
            health.effective_reader_fps,
            health.source_fps,
            health.loop_count,
            f" error={health.last_error}" if health.last_error else "",
        )


def show_preview(manager: CameraManager) -> bool:
    import cv2
    import numpy as np

    tiles = []
    for worker in manager.workers.values():
        packet = worker.latest_frame.get_latest(timeout=0.0)
        if packet is None:
            tile = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(tile, f"{worker.config.camera_id}: waiting", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        else:
            tile = cv2.resize(packet.image, (480, 270))
            age_ms = (time.monotonic() - packet.captured_monotonic) * 1000
            lines = (
                f"{packet.camera_id} | {worker.config.zone_id}",
                f"frame {packet.frame_id} | source {packet.source_timestamp_ms:.0f} ms",
                f"age {age_ms:.0f} ms | loop {packet.loop_count}",
            )
            for line_number, line in enumerate(lines, start=1):
                cv2.putText(tile, line, (10, line_number * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        tiles.append(tile)
    if not tiles:
        return False
    preview = np.hstack(tiles)
    cv2.imshow("Industrial Safety - Camera Preview", preview)
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
    configs = load_camera_configs(args.config, project_root=PROJECT_ROOT)
    manager = CameraManager(configs)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    next_stats_at = started_at

    try:
        manager.start_all()
        while not stop_requested.is_set():
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            if now >= next_stats_at:
                log_health(manager)
                next_stats_at = now + args.stats_interval
            if args.display:
                if show_preview(manager):
                    break
                stop_requested.wait(0.01)
            else:
                stop_requested.wait(min(0.25, max(0.01, next_stats_at - now)))
    finally:
        manager.stop_all()
        if args.display:
            import cv2

            cv2.destroyAllWindows()
        log_health(manager)

    errors = [health for health in manager.health().values() if health.state is CameraState.ERROR]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

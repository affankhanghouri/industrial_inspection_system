#!/usr/bin/env python3
"""Save representative configured-zone previews for CAM-002 and CAM-003."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from industrial_safety.video import load_camera_configs
from industrial_safety.zones import load_zones


def main() -> int:
    output = PROJECT_ROOT / "debug/zones"
    output.mkdir(parents=True, exist_ok=True)
    cameras = {
        camera.camera_id: camera
        for camera in load_camera_configs(project_root=PROJECT_ROOT)
    }
    source_frames = {"CAM-002": 180, "CAM-003": 270}
    for zone in load_zones(project_root=PROJECT_ROOT):
        camera = cameras[zone.camera_id]
        capture = cv2.VideoCapture(str(camera.source_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frames[zone.camera_id])
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            raise RuntimeError(f"could not decode {zone.camera_id} preview frame")
        points = np.asarray(zone.polygon, dtype=np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], (0, 165, 255))
        image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
        cv2.polylines(image, [points], True, (0, 165, 255), 4)
        cv2.rectangle(image, (0, 0), (image.shape[1], 48), (20, 24, 30), -1)
        cv2.putText(
            image,
            f"{zone.camera_id} | {zone.name} | bottom-center entry",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = output / f"{zone.camera_id.lower()}_zone.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

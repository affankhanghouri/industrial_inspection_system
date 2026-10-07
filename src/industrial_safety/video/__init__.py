"""Live-style video ingestion primitives."""

from .camera_manager import CameraManager
from .camera_worker import CameraWorker
from .config import CameraConfig, load_camera_configs
from .latest_frame import LatestFrameBuffer
from .mp4_source import MP4VideoSource
from .source import VideoSource
from .types import CameraHealth, CameraState, FramePacket, SourceFrame

__all__ = [
    "CameraConfig",
    "CameraHealth",
    "CameraManager",
    "CameraState",
    "CameraWorker",
    "FramePacket",
    "LatestFrameBuffer",
    "MP4VideoSource",
    "SourceFrame",
    "VideoSource",
    "load_camera_configs",
]

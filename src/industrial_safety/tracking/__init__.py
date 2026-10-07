"""Camera-local, latest-result person tracking."""

from .config import (
    TrackerConfig,
    TrackingConfig,
    TrackingRuntimeConfig,
    load_tracking_config,
)
from .latest_result import LatestTrackingResult
from .manager import CameraTrackingProcessor, TrackingCoordinator
from .time_aware_iou_tracker import TimeAwareIoUPersonTracker
from .tracker import PersonTracker
from .types import (
    CameraTrackingStats,
    GlobalTrackingStats,
    PersonTrack,
    TrackingResult,
    TrackState,
)

__all__ = [
    "CameraTrackingProcessor",
    "CameraTrackingStats",
    "GlobalTrackingStats",
    "LatestTrackingResult",
    "PersonTrack",
    "PersonTracker",
    "TimeAwareIoUPersonTracker",
    "TrackerConfig",
    "TrackingConfig",
    "TrackingCoordinator",
    "TrackingResult",
    "TrackingRuntimeConfig",
    "TrackState",
    "load_tracking_config",
]

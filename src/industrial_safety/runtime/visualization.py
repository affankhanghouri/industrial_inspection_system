"""Collision-aware OpenCV rendering of existing final-pipeline results."""

from __future__ import annotations

import cv2
import numpy as np

from industrial_safety.events.types import SafetyEvent
from industrial_safety.ppe.types import PPEAssociationResult, PPEObservationState
from industrial_safety.tracking.types import PersonTrack, TrackingResult, TrackState
from industrial_safety.zones.types import Zone, ZoneEvaluation, ZoneState

LabelRect = tuple[int, int, int, int]

FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECT_FONT_SCALE = 0.48
HUD_FONT_SCALE = 0.50
EVENT_FONT_SCALE = 0.66
OBJECT_THICKNESS = 1
HUD_THICKNESS = 1
EVENT_THICKNESS = 2
LABEL_PADDING = 4
EVENT_BANNER_HEIGHT = 50


def _label_size(
    text: str,
    *,
    font_scale: float = OBJECT_FONT_SCALE,
    thickness: int = OBJECT_THICKNESS,
) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    return (
        width + 2 * LABEL_PADDING,
        height + baseline + 2 * LABEL_PADDING,
    )


def _overlaps(first: LabelRect, second: LabelRect) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _clamp_label(
    x: int,
    y: int,
    label_size: tuple[int, int],
    frame_size: tuple[int, int],
    reserved_top: int,
) -> LabelRect:
    frame_height, frame_width = frame_size
    width = min(label_size[0], frame_width)
    height = min(label_size[1], max(1, frame_height - reserved_top))
    left = max(0, min(x, frame_width - width))
    top = max(reserved_top, min(y, frame_height - height))
    return (left, top, left + width, top + height)


def find_non_overlapping_label_position(
    preferred_anchor: LabelRect,
    label_size: tuple[int, int],
    occupied_rects: list[LabelRect],
    frame_size: tuple[int, int],
    *,
    reserved_top: int = 0,
) -> LabelRect:
    """Place a label near an object without crossing prior labels or bounds."""

    x1, y1, x2, y2 = preferred_anchor
    label_width, label_height = label_size
    candidates = [
        (x1, y1 - label_height),  # above object
        (x1, y2),  # below object
        (x1, y1),  # top-left inside object
        (x2 - label_width, y1),  # top-right inside object
    ]
    # If the immediate choices collide, walk vertically by one label line.
    maximum_steps = max(1, frame_size[0] // max(1, label_height))
    for step in range(1, maximum_steps + 1):
        candidates.extend(
            (
                (x1, y1 - label_height - step * label_height),
                (x1, y2 + step * label_height),
            )
        )

    checked: set[LabelRect] = set()
    for x, y in candidates:
        candidate = _clamp_label(
            round(x), round(y), label_size, frame_size, reserved_top
        )
        if candidate in checked:
            continue
        checked.add(candidate)
        if not any(_overlaps(candidate, occupied) for occupied in occupied_rects):
            return candidate

    # A fully occupied frame is unusual; return a bounded deterministic fallback.
    return _clamp_label(x1, y1, label_size, frame_size, reserved_top)


def draw_label_with_background(
    canvas: np.ndarray,
    text: str,
    rect: LabelRect,
    *,
    background: tuple[int, int, int],
    foreground: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = OBJECT_FONT_SCALE,
    thickness: int = OBJECT_THICKNESS,
) -> None:
    """Draw one readable label inside its precomputed rectangle."""

    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), background, -1)
    (_, text_height), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.putText(
        canvas,
        text,
        (rect[0] + LABEL_PADDING, rect[1] + LABEL_PADDING + text_height),
        FONT,
        font_scale,
        foreground,
        thickness,
        cv2.LINE_AA,
    )


def _draw_label_group(
    canvas: np.ndarray,
    lines: tuple[tuple[str, tuple[int, int, int]], ...],
    object_rect: LabelRect,
    occupied_rects: list[LabelRect],
    *,
    reserved_top: int,
) -> LabelRect:
    sizes = tuple(_label_size(text) for text, _ in lines)
    group_size = (max(size[0] for size in sizes), sum(size[1] for size in sizes))
    group_rect = find_non_overlapping_label_position(
        object_rect,
        group_size,
        occupied_rects,
        canvas.shape[:2],
        reserved_top=reserved_top,
    )
    top = group_rect[1]
    for (text, background), (_, height) in zip(lines, sizes, strict=True):
        line_rect = (group_rect[0], top, group_rect[2], top + height)
        draw_label_with_background(
            canvas, text, line_rect, background=background
        )
        top += height
    occupied_rects.append(group_rect)
    return group_rect


def draw_person_track(
    canvas: np.ndarray,
    track: PersonTrack,
    occupied_rects: list[LabelRect],
    *,
    secondary_label: str | None = None,
    secondary_color: tuple[int, int, int] = (70, 70, 70),
    event_track: bool = False,
    reserved_top: int = 0,
) -> None:
    """Draw one person with a compact track label and optional state line."""

    if event_track:
        box_color = (0, 0, 255)
    elif track.state is TrackState.LOST:
        box_color = (110, 110, 110)
    else:
        box_color = (40, 210, 70)
    object_rect = (
        round(track.x1),
        round(track.y1),
        round(track.x2),
        round(track.y2),
    )
    cv2.rectangle(
        canvas,
        (object_rect[0], object_rect[1]),
        (object_rect[2], object_rect[3]),
        box_color,
        2,
    )
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"T{track.track_id}", (40, 90, 150))
    ]
    if secondary_label:
        lines.append((secondary_label, secondary_color))
    _draw_label_group(
        canvas,
        tuple(lines),
        object_rect,
        occupied_rects,
        reserved_top=reserved_top,
    )


def draw_helmet_observation(
    canvas: np.ndarray,
    observation: object,
    occupied_rects: list[LabelRect],
    *,
    reserved_top: int = 0,
) -> None:
    """Draw one raw helmet box and its compact detector confidence."""

    object_rect = (
        round(observation.x1),
        round(observation.y1),
        round(observation.x2),
        round(observation.y2),
    )
    cv2.rectangle(
        canvas,
        (object_rect[0], object_rect[1]),
        (object_rect[2], object_rect[3]),
        (0, 220, 255),
        2,
    )
    text = f"Helmet {observation.confidence:.2f}"
    label_rect = find_non_overlapping_label_position(
        object_rect,
        _label_size(text),
        occupied_rects,
        canvas.shape[:2],
        reserved_top=reserved_top,
    )
    draw_label_with_background(
        canvas,
        text,
        label_rect,
        background=(25, 125, 145),
    )
    occupied_rects.append(label_rect)


def draw_zone(
    canvas: np.ndarray,
    zone: Zone,
    occupied_rects: list[LabelRect],
    *,
    reserved_top: int = 0,
) -> None:
    points = np.asarray(zone.polygon, dtype=np.int32)
    cv2.polylines(canvas, [points], True, (0, 165, 255), 3)
    left = int(points[:, 0].min())
    top = int(points[:, 1].min())
    right = int(points[:, 0].max())
    bottom = int(points[:, 1].max())
    text = zone.name.upper().replace(" ENTRY", "")
    label_rect = find_non_overlapping_label_position(
        (left, top, right, bottom),
        _label_size(text),
        occupied_rects,
        canvas.shape[:2],
        reserved_top=reserved_top,
    )
    draw_label_with_background(
        canvas,
        text,
        label_rect,
        background=(120, 80, 10),
    )
    occupied_rects.append(label_rect)


def draw_hud(
    canvas: np.ndarray,
    *,
    camera_id: str,
    camera_fps: float,
    person_fps: float,
    ppe_fps: float | None,
    track_count: int,
    person_age_ms: float | None,
    ppe_age_ms: float | None,
) -> LabelRect:
    lines = [
        camera_id,
        f"Camera: {camera_fps:.1f} FPS",
        f"Person AI: {person_fps:.1f} FPS",
        f"PPE: {ppe_fps:.1f} FPS" if ppe_fps is not None else "PPE: disabled",
        f"Tracks: {track_count}",
    ]
    ages = []
    if person_age_ms is not None:
        ages.append(f"Person age {person_age_ms:.0f} ms")
    if ppe_age_ms is not None:
        ages.append(f"PPE age {ppe_age_ms:.0f} ms")
    if ages:
        lines.append(" | ".join(ages))
    sizes = tuple(
        _label_size(text, font_scale=HUD_FONT_SCALE, thickness=HUD_THICKNESS)
        for text in lines
    )
    width = max(size[0] for size in sizes) + 8
    line_height = max(size[1] for size in sizes)
    height = line_height * len(lines) + 8
    left = 10
    top = max(0, canvas.shape[0] - height - 10)
    rect = (left, top, min(canvas.shape[1], left + width), top + height)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (25, 28, 34), -1)
    y = top + 4
    for index, text in enumerate(lines):
        (_, text_height), _ = cv2.getTextSize(
            text, FONT, HUD_FONT_SCALE, HUD_THICKNESS
        )
        color = (0, 220, 255) if index == 0 else (240, 240, 240)
        cv2.putText(
            canvas,
            text,
            (left + 8, y + LABEL_PADDING + text_height),
            FONT,
            HUD_FONT_SCALE,
            color,
            HUD_THICKNESS,
            cv2.LINE_AA,
        )
        y += line_height
    return rect


def draw_event_banner(canvas: np.ndarray, event: SafetyEvent) -> LabelRect:
    rect = (0, 0, canvas.shape[1], EVENT_BANNER_HEIGHT)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 175), -1)
    text = f"CONFIRMED: {event.event_type} | TRACK #{event.track_id}"
    (_, text_height), _ = cv2.getTextSize(
        text, FONT, EVENT_FONT_SCALE, EVENT_THICKNESS
    )
    cv2.putText(
        canvas,
        text,
        (12, (EVENT_BANNER_HEIGHT + text_height) // 2),
        FONT,
        EVENT_FONT_SCALE,
        (255, 255, 255),
        EVENT_THICKNESS,
        cv2.LINE_AA,
    )
    return rect


def draw_safety_preview(
    image: np.ndarray,
    tracking: TrackingResult | None,
    *,
    zones: tuple[Zone, ...] = (),
    zone_evaluation: ZoneEvaluation | None = None,
    ppe_result: PPEAssociationResult | None = None,
    latest_event: SafetyEvent | None = None,
    camera_id: str | None = None,
    camera_fps: float = 0.0,
    person_fps: float = 0.0,
    ppe_fps: float | None = None,
    person_age_ms: float | None = None,
    ppe_age_ms: float | None = None,
) -> np.ndarray:
    """Render latest camera imagery with independently updated AI overlays."""

    canvas = image.copy()
    occupied_rects: list[LabelRect] = []
    reserved_top = EVENT_BANNER_HEIGHT if latest_event is not None else 0
    if latest_event is not None:
        occupied_rects.append((0, 0, canvas.shape[1], EVENT_BANNER_HEIGHT))

    resolved_camera_id = camera_id
    if resolved_camera_id is None and tracking is not None:
        resolved_camera_id = tracking.camera_id
    if resolved_camera_id is None and ppe_result is not None:
        resolved_camera_id = ppe_result.camera_id
    resolved_camera_id = resolved_camera_id or "CAMERA"

    tracks = () if tracking is None else tracking.tracks
    occupied_rects.append(
        draw_hud(
            canvas,
            camera_id=resolved_camera_id,
            camera_fps=camera_fps,
            person_fps=person_fps,
            ppe_fps=ppe_fps,
            track_count=sum(track.state is not TrackState.LOST for track in tracks),
            person_age_ms=person_age_ms,
            ppe_age_ms=ppe_age_ms,
        )
    )

    ppe_states: dict[int, PPEObservationState] = {}
    if ppe_result is not None:
        ppe_states = {
            item.track_id: item.state for item in ppe_result.track_observations
        }
    zone_states: dict[int, ZoneState] = {}
    if zone_evaluation is not None:
        zone_states = {
            item.track.track_id: item.state
            for item in zone_evaluation.observations
        }

    event_track_id = None if latest_event is None else latest_event.track_id
    for track in tracks:
        secondary_label = None
        secondary_color = (70, 70, 70)
        if (
            latest_event is not None
            and latest_event.event_type == "HELMET_MISSING"
            and track.track_id == latest_event.track_id
        ):
            secondary_label = "Helmet MISSING"
            secondary_color = (0, 0, 175)
        elif ppe_states.get(track.track_id) is PPEObservationState.OBSERVED:
            secondary_label = "Helmet OK"
            secondary_color = (30, 125, 40)
        elif ppe_states.get(track.track_id) is PPEObservationState.UNOBSERVED:
            secondary_label = "Helmet Missing"
            secondary_color = (0, 0, 175)
        elif track.track_id in zone_states:
            state = zone_states[track.track_id]
            secondary_label = state.value
            secondary_color = (
                (0, 0, 175) if state is ZoneState.INSIDE else (70, 70, 70)
            )
        draw_person_track(
            canvas,
            track,
            occupied_rects,
            secondary_label=secondary_label,
            secondary_color=secondary_color,
            event_track=track.track_id == event_track_id,
            reserved_top=reserved_top,
        )

    for zone in zones:
        draw_zone(
            canvas,
            zone,
            occupied_rects,
            reserved_top=reserved_top,
        )

    if ppe_result is not None:
        for observation in ppe_result.ppe_inference_result.observations:
            draw_helmet_observation(
                canvas,
                observation,
                occupied_rects,
                reserved_top=reserved_top,
            )

    # Draw last so no lower-priority geometry can obscure the event message.
    if latest_event is not None:
        draw_event_banner(canvas, latest_event)
    return canvas

"""Heuristic motion scoring for eating/drinking detection."""

import contextlib
import logging
import math
import os
import sys
import time
from collections import deque

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import mediapipe as mp
except Exception:
    mp = None

from dfx.constants import (
    CONSUMPTION_CLASS_NAMES,
    DRINK_CONTAINER_CLASS_NAMES,
    HAND_MOUTH_APPROACH_WINDOW_SECONDS,
    HAND_MOUTH_HOLD_SECONDS,
    HAND_MOUTH_LANDMARK_EMA_ALPHA,
    HAND_MOUTH_LANDMARK_MIN_DETECTION_CONFIDENCE,
    HAND_MOUTH_LANDMARK_MIN_TRACKING_CONFIDENCE,
    HAND_MOUTH_MAX_DISTANCE_RATIO,
    HAND_MOUTH_MAX_TRACK_GAP_SECONDS,
    HAND_MOUTH_MIN_DIRECTION_COSINE,
    HAND_MOUTH_MIN_APPROACH_DELTA_RATIO,
    HAND_MOUTH_MIN_BBOX_SCALE_PX,
    HAND_MOUTH_MIN_DWELL_SECONDS,
    HAND_MOUTH_MIN_PERSON_AREA_RATIO,
    HAND_MOUTH_MIN_PERSON_CONFIDENCE,
    HAND_MOUTH_MIN_WRIST_UPWARD_RATIO,
    HAND_MOUTH_MIN_WRIST_UPWARD_STEPS,
    HAND_MOUTH_PERSON_CROP_MARGIN_RATIO,
    HAND_MOUTH_SCORE_FLOOR,
    HAND_MOUTH_WRIST_UPWARD_HISTORY_SECONDS,
    HANDHELD_FOOD_CLASS_NAMES,
    MOTION_TRIGGER_SCORE,
)

logger = logging.getLogger(__name__)

_POSE_MOUTH_INDICES = (9, 10)
_POSE_WRIST_INDICES = (15, 16)


def _extract_detection_geometry(det: dict) -> tuple[float, float, float] | None:
    """Return center and diagonal size for a detection, or None when incomplete."""
    bbox = det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    bbox_center_x = x1 + (width * 0.5)
    bbox_center_y = y1 + (height * 0.5)

    # Top-down cameras can produce skewed/squashed boxes, so use a blended size metric
    # instead of relying only on the diagonal.
    box_diag = math.hypot(width, height)
    box_area_side = math.sqrt(width * height)
    box_scale = max(1.0, (box_area_side * 0.65) + (box_diag * 0.35))

    center = det.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        try:
            det_center_x = float(center[0])
            det_center_y = float(center[1])
        except (TypeError, ValueError):
            det_center_x = bbox_center_x
            det_center_y = bbox_center_y
        x = (det_center_x * 0.6) + (bbox_center_x * 0.4)
        y = (det_center_y * 0.6) + (bbox_center_y * 0.4)
    else:
        x, y = bbox_center_x, bbox_center_y

    return x, y, box_scale


def _consumption_track_key(class_name: str) -> str:
    """Group classes that often flicker between similar labels across nearby frames."""
    normalized = class_name.strip().lower()
    if normalized in DRINK_CONTAINER_CLASS_NAMES:
        return "drink_container"
    if normalized in HANDHELD_FOOD_CLASS_NAMES:
        return "handheld_food"
    return normalized


def _smooth_motion_history(history: list[tuple[float, float, float, float]]):
    """Reduce detector jitter so motion scoring reacts to the actual trajectory."""
    if not history:
        return []
    smoothed: list[tuple[float, float, float, float]] = [history[0]]
    alpha = 0.45
    prev_x, prev_y, prev_diag, _ = history[0]
    for x, y, diag, ts in history[1:]:
        prev_x = (prev_x * (1.0 - alpha)) + (x * alpha)
        prev_y = (prev_y * (1.0 - alpha)) + (y * alpha)
        prev_diag = (prev_diag * (1.0 - alpha)) + (diag * alpha)
        smoothed.append((prev_x, prev_y, prev_diag, ts))
    return smoothed


def _extract_person_anchor(det: dict) -> tuple[float, float, float, float, float, float] | None:
    """Estimate a rough mouth-area target from one person bounding box."""
    bbox = det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + (width * 0.5)
    mouth_y = y1 + (height * 0.28)
    radius = max(30.0, min(width, height) * 0.22)
    return center_x, mouth_y, radius, x1, y1, x2, y2


def reset_person_hand_to_mouth_state(config):
    """Reset the landmark-based hand-to-mouth state for one camera worker."""
    config.person_proxy_prev_gray = None
    config.person_proxy_active_until = 0.0
    config.person_proxy_trigger_streak = 0
    config.person_proxy_dwell_started_at = 0.0
    config.person_proxy_last_seen_ts = 0.0
    config.person_proxy_last_approach_ts = 0.0
    config.person_proxy_last_distance_ratio = float("inf")
    config.person_proxy_subject_bbox = None
    config.person_proxy_last_wrist_xy = None
    config.person_proxy_last_finger_xy = None
    config.person_proxy_last_mouth_xy = None
    config.person_proxy_wrist_history = deque(maxlen=16)


def _current_hand_mouth_subject(config):
    """Return the latest proxy subject geometry used by camera-level person tracking."""
    subject_bbox = getattr(config, "person_proxy_subject_bbox", None)
    wrist_xy = getattr(config, "person_proxy_last_wrist_xy", None)
    if wrist_xy is None:
        wrist_xy = getattr(config, "person_proxy_last_finger_xy", None)
    mouth_xy = getattr(config, "person_proxy_last_mouth_xy", None)
    if subject_bbox is None and wrist_xy is None and mouth_xy is None:
        return None
    payload: dict[str, object] = {}
    if isinstance(subject_bbox, (list, tuple)) and len(subject_bbox) == 4:
        payload["subject_bbox_xyxy"] = [float(v) for v in subject_bbox]
    if isinstance(wrist_xy, (list, tuple)) and len(wrist_xy) == 2:
        payload["wrist_xy"] = [float(wrist_xy[0]), float(wrist_xy[1])]
    if isinstance(mouth_xy, (list, tuple)) and len(mouth_xy) == 2:
        payload["mouth_xy"] = [float(mouth_xy[0]), float(mouth_xy[1])]
    return payload or None


def _get_hand_mouth_detector(config):
    """Lazily create the MediaPipe holistic detector used for strict landmark checks."""
    detector = getattr(config, "person_proxy_landmark_detector", None)
    if detector is not None:
        return detector
    if bool(getattr(config, "person_proxy_landmark_detector_unavailable", False)):
        return None
    if mp is None:
        logger.warning("mediapipe unavailable; strict hand-to-mouth landmark detection disabled")
        config.person_proxy_landmark_detector_unavailable = True
        return None
    try:
        with _suppress_native_stderr():
            detector = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=HAND_MOUTH_LANDMARK_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_MOUTH_LANDMARK_MIN_TRACKING_CONFIDENCE,
            )
    except Exception as exc:
        logger.warning("could not initialize mediapipe holistic detector: %s", exc)
        config.person_proxy_landmark_detector_unavailable = True
        return None
    config.person_proxy_landmark_detector = detector
    config.person_proxy_detector_quiet_frames = 4
    return detector


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence one-time MediaPipe/TFLite constructor noise written directly to stderr."""
    if os.name != "posix":
        yield
        return

    saved_fd = None
    devnull = None
    try:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved_fd = os.dup(2)
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if saved_fd is not None:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        if devnull is not None:
            devnull.close()


def _current_hand_mouth_output(config, now_ts: float) -> tuple[bool, float, dict | None]:
    """Return the currently active strict hand-to-mouth state and score."""
    active_until = float(getattr(config, "person_proxy_active_until", 0.0))
    active = active_until > 0.0 and now_ts <= active_until
    return active, (round(HAND_MOUTH_SCORE_FLOOR, 3) if active else 0.0), _current_hand_mouth_subject(config)


def _reset_hand_mouth_candidate(config):
    """Clear in-progress landmark proximity state without clearing an active hold."""
    config.person_proxy_dwell_started_at = 0.0
    config.person_proxy_last_seen_ts = 0.0
    config.person_proxy_last_approach_ts = 0.0
    config.person_proxy_last_distance_ratio = float("inf")
    config.person_proxy_subject_bbox = None
    config.person_proxy_last_wrist_xy = None
    config.person_proxy_last_finger_xy = None
    config.person_proxy_last_mouth_xy = None
    config.person_proxy_wrist_history = deque(maxlen=16)


def _handle_hand_mouth_gap(config, now_ts: float) -> tuple[bool, float, dict | None]:
    """Gracefully expire proximity state after landmark loss or missing detections."""
    last_seen_ts = float(getattr(config, "person_proxy_last_seen_ts", 0.0))
    if last_seen_ts <= 0.0 or (now_ts - last_seen_ts) > HAND_MOUTH_MAX_TRACK_GAP_SECONDS:
        _reset_hand_mouth_candidate(config)
    return _current_hand_mouth_output(config, now_ts)


def _safe_landmark_xy(
    landmarks,
    index: int,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
    pad_x: int = 0,
    pad_y: int = 0,
):
    """Project one MediaPipe landmark index into full-frame pixel coordinates."""
    if landmarks is None:
        return None
    points = getattr(landmarks, "landmark", None)
    if points is None or index >= len(points):
        return None
    point = points[index]
    raw_x = (float(point.x) * float(width)) - float(pad_x)
    raw_y = (float(point.y) * float(height)) - float(pad_y)
    return (
        float(offset_x) + raw_x,
        float(offset_y) + raw_y,
    )


def _mean_landmark_xy(
    landmarks,
    indices: tuple[int, ...],
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
    pad_x: int = 0,
    pad_y: int = 0,
):
    """Project multiple landmarks and return their mean pixel location."""
    coords = [
        _safe_landmark_xy(landmarks, index, offset_x, offset_y, width, height, pad_x, pad_y)
        for index in indices
    ]
    coords = [coord for coord in coords if coord is not None]
    if not coords:
        return None
    return (
        sum(coord[0] for coord in coords) / float(len(coords)),
        sum(coord[1] for coord in coords) / float(len(coords)),
    )


def _distance(point_a, point_b) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def _ema_point(previous_xy, current_xy, alpha: float):
    """Lightweight EMA smoothing for landmark coordinates."""
    if current_xy is None:
        return previous_xy
    if previous_xy is None:
        return current_xy
    smooth_alpha = max(0.05, min(0.95, float(alpha)))
    return (
        float(previous_xy[0]) + ((float(current_xy[0]) - float(previous_xy[0])) * smooth_alpha),
        float(previous_xy[1]) + ((float(current_xy[1]) - float(previous_xy[1])) * smooth_alpha),
    )


def _direction_cosine(motion_vector: tuple[float, float], target_vector: tuple[float, float]) -> float:
    """Return cosine similarity between movement and mouth-target vectors."""
    motion_norm = math.hypot(motion_vector[0], motion_vector[1])
    target_norm = math.hypot(target_vector[0], target_vector[1])
    if motion_norm <= 1e-6 or target_norm <= 1e-6:
        return 0.0
    return (
        (motion_vector[0] * target_vector[0]) + (motion_vector[1] * target_vector[1])
    ) / (motion_norm * target_norm)


def _select_person_crop(person_detections: list[dict], frame_width: int, frame_height: int):
    """Pick the strongest person ROI so landmark inference focuses on one subject."""
    best_bbox = None
    best_score = 0.0
    frame_area = max(1.0, float(frame_width * frame_height))
    for det in person_detections:
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < HAND_MOUTH_MIN_PERSON_CONFIDENCE:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        if (width * height) / frame_area < HAND_MOUTH_MIN_PERSON_AREA_RATIO:
            continue
        score = (width * height) * max(0.1, confidence)
        if score > best_score:
            best_score = score
            best_bbox = (x1, y1, x2, y2)
    if best_bbox is None:
        return None
    x1, y1, x2, y2 = best_bbox
    margin_x = (x2 - x1) * HAND_MOUTH_PERSON_CROP_MARGIN_RATIO
    margin_y = (y2 - y1) * HAND_MOUTH_PERSON_CROP_MARGIN_RATIO
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(frame_width, int(round(x2 + margin_x)))
    bottom = min(frame_height, int(round(y2 + margin_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _select_hand_mouth_crop(person_detections: list[dict], frame_width: int, frame_height: int):
    """Use the strongest person crop when available, otherwise fall back to the full frame."""
    person_crop = _select_person_crop(person_detections, frame_width, frame_height)
    if person_crop is not None:
        return person_crop
    if frame_width <= 0 or frame_height <= 0:
        return None
    return 0, 0, int(frame_width), int(frame_height)


def _prepare_hand_mouth_input(frame, crop_bounds: tuple[int, int, int, int]):
    """Square-pad the selected crop before MediaPipe so its ROI math stays stable."""
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop = frame[crop_top:crop_bottom, crop_left:crop_right]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    input_frame = crop
    pad_x = 0
    pad_y = 0
    if crop_w != crop_h:
        side = max(crop_w, crop_h)
        input_frame = np.zeros((side, side, 3), dtype=crop.dtype)
        pad_x = (side - crop_w) // 2
        pad_y = (side - crop_h) // 2
        input_frame[pad_y:pad_y + crop_h, pad_x:pad_x + crop_w] = crop
    return {
        "frame": input_frame,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def _hand_mouth_score(
    distance_ratio: float,
    dwell_elapsed: float,
    approach_delta_ratio: float,
    direction_cosine: float,
    wrist_upward_ratio: float,
    velocity_ready: bool,
    active: bool,
) -> float:
    """Convert strict landmark geometry into a bounded score used by the alert pipeline."""
    proximity_score = max(0.0, 1.0 - (distance_ratio / max(1e-6, HAND_MOUTH_MAX_DISTANCE_RATIO)))
    dwell_score = min(1.0, dwell_elapsed / max(1e-6, HAND_MOUTH_MIN_DWELL_SECONDS))

    if HAND_MOUTH_MIN_APPROACH_DELTA_RATIO > 1e-6:
        approach_score = min(1.0, max(0.0, approach_delta_ratio / HAND_MOUTH_MIN_APPROACH_DELTA_RATIO))
    else:
        approach_score = 1.0 if approach_delta_ratio > 0.0 else 0.0

    if HAND_MOUTH_MIN_DIRECTION_COSINE <= 0.0:
        direction_score = min(1.0, max(0.0, (direction_cosine + 1.0) * 0.5))
    elif HAND_MOUTH_MIN_DIRECTION_COSINE >= 1.0:
        direction_score = 1.0 if direction_cosine >= HAND_MOUTH_MIN_DIRECTION_COSINE else 0.0
    else:
        direction_score = min(
            1.0,
            max(
                0.0,
                (direction_cosine - HAND_MOUTH_MIN_DIRECTION_COSINE)
                / max(1e-6, 1.0 - HAND_MOUTH_MIN_DIRECTION_COSINE),
            ),
        )
    upward_score = min(
        1.0,
        max(0.0, wrist_upward_ratio / max(1e-6, HAND_MOUTH_MIN_WRIST_UPWARD_RATIO)),
    )
    if velocity_ready:
        upward_score = max(upward_score, 1.0)

    very_close = distance_ratio <= (HAND_MOUTH_MAX_DISTANCE_RATIO * 0.55)
    if very_close:
        proximity_weight = 0.58
        dwell_weight = 0.20
        approach_weight = 0.05
        direction_weight = 0.03
        upward_weight = 0.14
    else:
        proximity_weight = 0.42
        dwell_weight = 0.25
        approach_weight = 0.12
        direction_weight = 0.09
        upward_weight = 0.12

    raw_score = (
        (proximity_weight * proximity_score)
        + (dwell_weight * dwell_score)
        + (approach_weight * approach_score)
        + (direction_weight * direction_score)
        + (upward_weight * upward_score)
    )
    score = min(1.5, max(0.0, raw_score * 1.45))
    if not velocity_ready and not active:
        score *= 0.72
    if active:
        score = max(score, HAND_MOUTH_SCORE_FLOOR)
    return round(score, 3)


def _wrist_upward_stats(config, wrist_xy, now_ts: float, person_height_px: float) -> tuple[float, int, bool]:
    """Track short-term wrist Y movement and require upward motion before triggering."""
    history = getattr(config, "person_proxy_wrist_history", None)
    if not isinstance(history, deque):
        history = deque(maxlen=16)
        config.person_proxy_wrist_history = history
    history.append((float(now_ts), float(wrist_xy[0]), float(wrist_xy[1])))
    while history and (now_ts - float(history[0][0])) > HAND_MOUTH_WRIST_UPWARD_HISTORY_SECONDS:
        history.popleft()
    if len(history) < 2:
        return 0.0, 0, False

    history_points = list(history)
    upward_pixels = float(history_points[0][2]) - float(history_points[-1][2])
    upward_ratio = upward_pixels / max(1.0, float(person_height_px))
    upward_steps = sum(
        1
        for prev, cur in zip(history_points, history_points[1:])
        if (float(prev[2]) - float(cur[2])) > 1.0
    )
    velocity_ready = (
        upward_ratio >= HAND_MOUTH_MIN_WRIST_UPWARD_RATIO
        and upward_steps >= HAND_MOUTH_MIN_WRIST_UPWARD_STEPS
    )
    return upward_ratio, upward_steps, velocity_ready


def _score_person_proximity(
    person_detections: list[dict],
    latest_point: tuple[float, float],
    peak_point: tuple[float, float],
) -> float:
    """Reward motions that end near a detected person's upper face/head area."""
    best_score = 0.0
    latest_x, latest_y = latest_point
    peak_x, peak_y = peak_point
    for det in person_detections:
        anchor = _extract_person_anchor(det)
        if anchor is None:
            continue
        mouth_x, mouth_y, radius, x1, y1, x2, y2 = anchor
        latest_distance = math.hypot(latest_x - mouth_x, latest_y - mouth_y)
        peak_distance = math.hypot(peak_x - mouth_x, peak_y - mouth_y)
        latest_score = max(0.0, 1.0 - (latest_distance / max(1.0, radius * 2.0)))
        peak_score = max(0.0, 1.0 - (peak_distance / max(1.0, radius * 2.0)))
        inside_upper_person = (
            x1 <= latest_x <= x2 and y1 <= latest_y <= (y1 + ((y2 - y1) * 0.45))
        )
        best_score = max(
            best_score,
            latest_score,
            peak_score,
            1.0 if inside_upper_person else 0.0,
        )
    return best_score


def _score_motion_track(
    track: dict,
    frame_diag: float,
    frame_height: int,
    config,
    person_detections: list[dict] | None = None,
) -> float:
    """Score a matched object track based on recent path, lift, and proximity to a person."""
    raw_history = list(track.get("history", ()))
    if len(raw_history) < 4:
        return 0.0
    history = _smooth_motion_history(raw_history)
    path_length = 0.0
    upward_total = 0.0
    downward_total = 0.0
    min_y = history[0][1]
    min_entry = history[0]
    rising_steps = 0
    for prev, cur in zip(history, history[1:]):
        path_length += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        delta_y = prev[1] - cur[1]
        upward_total += max(0.0, delta_y)
        downward_total += max(0.0, -delta_y)
        if delta_y > max(2.0, frame_height * 0.006):
            rising_steps += 1
        if cur[1] < min_y:
            min_y = cur[1]
            min_entry = cur
    first_x, first_y, first_diag, _ = history[0]
    last_x, last_y, _, _ = history[-1]
    net_displacement = math.hypot(last_x - first_x, last_y - first_y)
    net_upward = max(0.0, first_y - last_y)
    horizontal_travel = abs(last_x - first_x)
    path_norm = path_length / max(1.0, frame_diag)
    displacement_norm = net_displacement / max(1.0, frame_diag)
    upward_norm = upward_total / max(1.0, float(frame_height))
    downward_norm = downward_total / max(1.0, float(frame_height))
    net_upward_norm = net_upward / max(1.0, float(frame_height))
    vertical_gain_norm = max(0.0, first_y - min_y) / max(1.0, float(frame_height))
    linearity = net_displacement / max(1.0, path_length)
    vertical_dominance = net_upward / max(1.0, horizontal_travel + net_upward)
    directional_consistency = rising_steps / max(1.0, float(len(history) - 1))
    size_growth = max(0.0, (max(point[2] for point in history) - first_diag) / max(1.0, first_diag))
    upper_zone_score = max(0.0, 0.72 - (min_y / max(1.0, float(frame_height)))) / 0.72
    person_proximity = _score_person_proximity(
        person_detections or [],
        latest_point=(last_x, last_y),
        peak_point=(min_entry[0], min_entry[1]),
    )

    # Hard gate: there should be meaningful object movement, or a clear approach toward a
    # detected person's head/face region, otherwise we are likely just seeing bbox jitter.
    if displacement_norm < (config.motion_displacement_threshold * 0.35) and person_proximity < 0.65:
        track.get("score_history", deque()).clear()
        return 0.0
    if vertical_gain_norm < (config.motion_upward_threshold * 0.45) and person_proximity < 0.55:
        track.get("score_history", deque()).clear()
        return 0.0
    if net_upward_norm < (config.motion_upward_threshold * 0.35) and person_proximity < 0.75:
        track.get("score_history", deque()).clear()
        return 0.0
    if linearity < 0.22 and directional_consistency < 0.35 and person_proximity < 0.75:
        track.get("score_history", deque()).clear()
        return 0.0
    if vertical_dominance < 0.18 and person_proximity < 0.7:
        track.get("score_history", deque()).clear()
        return 0.0

    displacement_score = displacement_norm / max(1e-6, config.motion_displacement_threshold)
    path_score = path_norm / max(1e-6, config.motion_displacement_threshold * 2.0)
    upward_score = upward_norm / max(1e-6, config.motion_upward_threshold)
    net_upward_score = net_upward_norm / max(1e-6, config.motion_upward_threshold)
    lift_score = vertical_gain_norm / max(1e-6, config.motion_upward_threshold * 1.25)
    downward_penalty = downward_norm / max(1e-6, config.motion_upward_threshold)
    size_growth_score = size_growth / 0.25

    raw_score = (
        0.16 * displacement_score
        + 0.08 * path_score
        + 0.16 * upward_score
        + 0.16 * net_upward_score
        + 0.12 * lift_score
        + 0.08 * linearity
        + 0.07 * directional_consistency
        + 0.05 * upper_zone_score
        + 0.05 * size_growth_score
        + 0.17 * person_proximity
        - 0.10 * downward_penalty
    )
    score_history = track.setdefault("score_history", deque(maxlen=max(3, config.motion_window // 3)))
    score_history.append(max(0.0, raw_score))
    return sum(score_history) / len(score_history)


def detect_consumption_motion(
    config,
    detections: list[dict],
    frame_width: int,
    frame_height: int,
    person_detections: list[dict] | None = None,
) -> tuple[bool, float]:
    """Heuristically score whether a detected item is moving like it is being consumed."""
    if not config.motion_enabled:
        return False, 0.0

    now = time.time()
    frame_diag = max(1.0, math.hypot(frame_width, frame_height))
    sample_fps = max(
        1.0,
        config.max_inference_fps if getattr(config, "max_inference_fps", 0.0) > 0.0 else config.stream_fps,
    )
    stale_after = max(2.0, config.motion_window / sample_fps)

    for track_id, track in list(config.motion_tracks.items()):
        if (now - float(track.get("last_seen", 0.0))) > stale_after:
            config.motion_tracks.pop(track_id, None)

    candidates: list[tuple[int, dict, str, float, float, float]] = []
    for det_index, det in enumerate(detections):
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name not in CONSUMPTION_CLASS_NAMES:
            continue
        geometry = _extract_detection_geometry(det)
        if geometry is None:
            continue
        x, y, box_diag = geometry
        candidates.append((det_index, det, _consumption_track_key(class_name), x, y, box_diag))

    used_track_ids: set[int] = set()
    matched_track_ids: dict[int, int] = {}
    for det_index, det, track_key, x, y, box_diag in sorted(
        candidates,
        key=lambda item: float(item[1].get("confidence", 0.0)),
        reverse=True,
    ):
        best_track_id = None
        best_distance = float("inf")
        max_match_distance = max(frame_diag * 0.05, box_diag * 1.6, 60.0)
        for track_id, track in config.motion_tracks.items():
            if track_id in used_track_ids:
                continue
            if track.get("track_key") != track_key:
                continue
            last_x, last_y, _, _ = track["history"][-1]
            distance = math.hypot(x - last_x, y - last_y)
            if distance <= max_match_distance and distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        if best_track_id is None:
            best_track_id = config.next_motion_track_id
            config.next_motion_track_id += 1
            config.motion_tracks[best_track_id] = {
                "track_key": track_key,
                "history": deque(maxlen=config.motion_window),
                "last_seen": now,
                "score_history": deque(maxlen=max(3, config.motion_window // 3)),
                "active_until": 0.0,
            }
        track = config.motion_tracks[best_track_id]
        track["last_seen"] = now
        track["history"].append((x, y, box_diag, now))
        used_track_ids.add(best_track_id)
        matched_track_ids[det_index] = best_track_id

    max_score = 0.0
    for det_index, det in enumerate(detections):
        track_id = matched_track_ids.get(det_index)
        if track_id is None:
            det["motion_score"] = 0.0
            det["consumption_motion"] = False
            continue
        track = config.motion_tracks.get(track_id)
        score = (
            0.0
            if track is None
            else _score_motion_track(
                track,
                frame_diag,
                frame_height,
                config,
                person_detections=person_detections,
            )
        )
        if track is not None and score >= MOTION_TRIGGER_SCORE:
            track["active_until"] = now + max(0.0, float(getattr(config, "motion_hold_seconds", 0.1)))
        effective_score = score
        if track is not None and now <= float(track.get("active_until", 0.0)):
            effective_score = max(effective_score, MOTION_TRIGGER_SCORE)
        det["motion_track_id"] = track_id
        det["motion_score"] = round(effective_score, 3)
        det["consumption_motion"] = bool(effective_score >= MOTION_TRIGGER_SCORE)
        if effective_score > max_score:
            max_score = effective_score

    return max_score >= MOTION_TRIGGER_SCORE, round(max_score, 3)


def detect_person_hand_to_mouth_proxy(
    config,
    frame,
    person_detections: list[dict],
    now_ts: float,
) -> tuple[bool, float, dict | None]:
    """Strict hand-to-mouth signal from pose wrists-to-mouth 2D proximity."""
    if cv2 is None or np is None:
        return False, 0.0, None

    detector = _get_hand_mouth_detector(config)
    if detector is None:
        return _current_hand_mouth_output(config, now_ts)

    frame_h, frame_w = frame.shape[:2]
    person_crop = _select_hand_mouth_crop(person_detections, frame_w, frame_h)
    if person_crop is None:
        return _handle_hand_mouth_gap(config, now_ts)
    prepared_input = _prepare_hand_mouth_input(frame, person_crop)
    if prepared_input is None:
        return _handle_hand_mouth_gap(config, now_ts)

    crop_left = int(prepared_input["crop_left"])
    crop_top = int(prepared_input["crop_top"])
    crop_right = int(person_crop[2])
    crop_bottom = int(person_crop[3])
    config.person_proxy_subject_bbox = [crop_left, crop_top, crop_right, crop_bottom]
    pad_x = int(prepared_input["pad_x"])
    pad_y = int(prepared_input["pad_y"])
    input_frame = prepared_input["frame"]
    input_h, input_w = input_frame.shape[:2]
    rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    quiet_frames = int(getattr(config, "person_proxy_detector_quiet_frames", 0))
    if quiet_frames > 0:
        with _suppress_native_stderr():
            results = detector.process(rgb)
        config.person_proxy_detector_quiet_frames = quiet_frames - 1
    else:
        results = detector.process(rgb)

    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return _handle_hand_mouth_gap(config, now_ts)

    mouth_center = _mean_landmark_xy(
        pose_landmarks,
        _POSE_MOUTH_INDICES,
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    if mouth_center is None:
        return _handle_hand_mouth_gap(config, now_ts)

    person_width = max(1.0, float(crop_right - crop_left))
    person_height = max(1.0, float(crop_bottom - crop_top))
    person_scale = max(HAND_MOUTH_MIN_BBOX_SCALE_PX, math.sqrt(person_width * person_height))
    if person_scale < HAND_MOUTH_MIN_BBOX_SCALE_PX:
        return _handle_hand_mouth_gap(config, now_ts)

    wrist_candidates = [
        _safe_landmark_xy(
            pose_landmarks,
            _POSE_WRIST_INDICES[0],
            crop_left,
            crop_top,
            input_w,
            input_h,
            pad_x,
            pad_y,
        ),
        _safe_landmark_xy(
            pose_landmarks,
            _POSE_WRIST_INDICES[1],
            crop_left,
            crop_top,
            input_w,
            input_h,
            pad_x,
            pad_y,
        ),
    ]
    wrist_candidates = [candidate for candidate in wrist_candidates if candidate is not None]
    if not wrist_candidates:
        return _handle_hand_mouth_gap(config, now_ts)

    closest_wrist_xy = None
    closest_distance_ratio = float("inf")
    for wrist_xy in wrist_candidates:
        distance_ratio = _distance(wrist_xy, mouth_center) / max(1.0, person_scale)
        if distance_ratio < closest_distance_ratio:
            closest_distance_ratio = distance_ratio
            closest_wrist_xy = wrist_xy
    if closest_wrist_xy is None:
        return _handle_hand_mouth_gap(config, now_ts)

    previous_wrist_xy = getattr(config, "person_proxy_last_wrist_xy", None)
    if previous_wrist_xy is None:
        previous_wrist_xy = getattr(config, "person_proxy_last_finger_xy", None)
    previous_mouth_xy = getattr(config, "person_proxy_last_mouth_xy", None)
    smooth_alpha = float(getattr(config, "person_proxy_landmark_ema_alpha", HAND_MOUTH_LANDMARK_EMA_ALPHA))
    mouth_center = _ema_point(previous_mouth_xy, mouth_center, smooth_alpha)
    closest_wrist_xy = _ema_point(previous_wrist_xy, closest_wrist_xy, smooth_alpha)
    if mouth_center is None or closest_wrist_xy is None:
        return _handle_hand_mouth_gap(config, now_ts)

    closest_distance_ratio = _distance(closest_wrist_xy, mouth_center) / max(1.0, person_scale)
    previous_distance_ratio = float(getattr(config, "person_proxy_last_distance_ratio", float("inf")))
    approach_delta_ratio = 0.0
    direction_cosine = 0.0
    if math.isfinite(previous_distance_ratio):
        approach_delta_ratio = max(0.0, previous_distance_ratio - closest_distance_ratio)
    if previous_wrist_xy is not None:
        motion_vector = (
            closest_wrist_xy[0] - previous_wrist_xy[0],
            closest_wrist_xy[1] - previous_wrist_xy[1],
        )
        target_origin = previous_mouth_xy if previous_mouth_xy is not None else mouth_center
        target_vector = (
            target_origin[0] - previous_wrist_xy[0],
            target_origin[1] - previous_wrist_xy[1],
        )
        direction_cosine = _direction_cosine(motion_vector, target_vector)

    wrist_upward_ratio, wrist_upward_steps, velocity_ready = _wrist_upward_stats(
        config,
        closest_wrist_xy,
        now_ts,
        person_height_px=person_height,
    )

    trajectory_ready = (
        approach_delta_ratio >= HAND_MOUTH_MIN_APPROACH_DELTA_RATIO
        or direction_cosine >= HAND_MOUTH_MIN_DIRECTION_COSINE
    )
    proximity_override = closest_distance_ratio <= (HAND_MOUTH_MAX_DISTANCE_RATIO * 1.10)
    if proximity_override or (
        closest_distance_ratio <= (HAND_MOUTH_MAX_DISTANCE_RATIO * 1.55)
        and trajectory_ready
        and velocity_ready
    ):
        config.person_proxy_last_approach_ts = now_ts

    recent_approach = (
        now_ts - float(getattr(config, "person_proxy_last_approach_ts", 0.0))
    ) <= HAND_MOUTH_APPROACH_WINDOW_SECONDS

    dwell_started_at = float(getattr(config, "person_proxy_dwell_started_at", 0.0))
    threshold_for_reset = HAND_MOUTH_MAX_DISTANCE_RATIO * (1.25 if dwell_started_at > 0.0 else 1.08)
    within_or_sticky = closest_distance_ratio <= threshold_for_reset
    if within_or_sticky:
        if dwell_started_at <= 0.0 and (recent_approach or proximity_override or velocity_ready):
            config.person_proxy_dwell_started_at = now_ts
    else:
        config.person_proxy_dwell_started_at = 0.0

    dwell_started_at = float(getattr(config, "person_proxy_dwell_started_at", 0.0))
    dwell_elapsed = max(0.0, now_ts - dwell_started_at) if dwell_started_at > 0.0 else 0.0
    triggered = (
        within_or_sticky
        and dwell_started_at > 0.0
        and dwell_elapsed >= HAND_MOUTH_MIN_DWELL_SECONDS
        and velocity_ready
        and recent_approach
    )
    if triggered:
        config.person_proxy_active_until = now_ts + HAND_MOUTH_HOLD_SECONDS
    active_until = float(getattr(config, "person_proxy_active_until", 0.0))
    active = active_until > 0.0 and now_ts <= active_until

    config.person_proxy_last_seen_ts = now_ts
    config.person_proxy_last_distance_ratio = closest_distance_ratio
    config.person_proxy_last_wrist_xy = closest_wrist_xy
    config.person_proxy_last_finger_xy = closest_wrist_xy
    config.person_proxy_last_mouth_xy = mouth_center

    score = _hand_mouth_score(
        distance_ratio=closest_distance_ratio,
        dwell_elapsed=dwell_elapsed,
        approach_delta_ratio=approach_delta_ratio,
        direction_cosine=direction_cosine,
        wrist_upward_ratio=wrist_upward_ratio,
        velocity_ready=velocity_ready,
        active=active,
    )
    proxy_details = {
        "subject_bbox_xyxy": [crop_left, crop_top, crop_right, crop_bottom],
        "wrist_xy": [float(closest_wrist_xy[0]), float(closest_wrist_xy[1])],
        "mouth_xy": [float(mouth_center[0]), float(mouth_center[1])],
        "distance_ratio": float(closest_distance_ratio),
        "wrist_upward_ratio": float(wrist_upward_ratio),
        "wrist_upward_steps": int(wrist_upward_steps),
        "velocity_ready": bool(velocity_ready),
    }
    return active, score, proxy_details

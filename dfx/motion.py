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
    HAND_MOUTH_LANDMARK_MIN_DETECTION_CONFIDENCE,
    HAND_MOUTH_LANDMARK_MIN_TRACKING_CONFIDENCE,
    HAND_MOUTH_MAX_DISTANCE_RATIO,
    HAND_MOUTH_MAX_TRACK_GAP_SECONDS,
    HAND_MOUTH_MIN_DIRECTION_COSINE,
    HAND_MOUTH_MIN_APPROACH_DELTA_RATIO,
    HAND_MOUTH_MIN_DWELL_SECONDS,
    HAND_MOUTH_MIN_FACE_WIDTH_PX,
    HAND_MOUTH_MIN_PERSON_AREA_RATIO,
    HAND_MOUTH_MIN_PERSON_CONFIDENCE,
    HAND_MOUTH_PERSON_CROP_MARGIN_RATIO,
    HAND_MOUTH_SCORE_FLOOR,
    HANDHELD_FOOD_CLASS_NAMES,
    MOTION_TRIGGER_SCORE,
)

logger = logging.getLogger(__name__)

_FACE_LIP_CENTER_INDICES = (13, 14, 78, 308)
_FACE_LIP_CORNER_INDICES = (61, 291)
_FACE_WIDTH_INDICES = (234, 454)
_INDEX_FINGER_TIP_INDEX = 8


def _extract_detection_geometry(det: dict) -> tuple[float, float, float] | None:
    """Return center and diagonal size for a detection, or None when incomplete."""
    center = det.get("center_xy")
    bbox = det.get("bbox_xyxy")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    box_diag = math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))
    return x, y, box_diag


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
    config.person_proxy_last_finger_xy = None
    config.person_proxy_last_mouth_xy = None


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


def _current_hand_mouth_output(config, now_ts: float) -> tuple[bool, float]:
    """Return the currently active strict hand-to-mouth state and score."""
    active_until = float(getattr(config, "person_proxy_active_until", 0.0))
    active = active_until > 0.0 and now_ts <= active_until
    return active, (round(HAND_MOUTH_SCORE_FLOOR, 3) if active else 0.0)


def _reset_hand_mouth_candidate(config):
    """Clear in-progress landmark proximity state without clearing an active hold."""
    config.person_proxy_dwell_started_at = 0.0
    config.person_proxy_last_seen_ts = 0.0
    config.person_proxy_last_approach_ts = 0.0
    config.person_proxy_last_distance_ratio = float("inf")
    config.person_proxy_last_finger_xy = None
    config.person_proxy_last_mouth_xy = None


def _handle_hand_mouth_gap(config, now_ts: float) -> tuple[bool, float]:
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


def _hand_mouth_score(distance_ratio: float, dwell_elapsed: float, approach_delta_ratio: float, direction_cosine: float, active: bool) -> float:
    """Convert strict landmark geometry into a bounded score used by the alert pipeline."""
    proximity_score = max(0.0, 1.0 - (distance_ratio / max(1e-6, HAND_MOUTH_MAX_DISTANCE_RATIO)))
    dwell_score = min(1.0, dwell_elapsed / max(1e-6, HAND_MOUTH_MIN_DWELL_SECONDS))
    approach_score = min(1.0, approach_delta_ratio / max(1e-6, HAND_MOUTH_MIN_APPROACH_DELTA_RATIO))
    direction_score = min(
        1.0,
        max(0.0, (direction_cosine - HAND_MOUTH_MIN_DIRECTION_COSINE) / max(1e-6, 1.0 - HAND_MOUTH_MIN_DIRECTION_COSINE)),
    )
    raw_score = (
        0.50 * proximity_score
        + 0.28 * dwell_score
        + 0.14 * approach_score
        + 0.08 * direction_score
    )
    score = min(1.5, max(0.0, raw_score * 1.5))
    if active:
        score = max(score, HAND_MOUTH_SCORE_FLOOR)
    return round(score, 3)


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
) -> tuple[bool, float]:
    """Strict hand-to-mouth signal from index-finger and lip landmark proximity."""
    if cv2 is None or np is None:
        return False, 0.0

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
    face_landmarks = getattr(results, "face_landmarks", None)
    hand_landmarks = [
        getattr(results, "left_hand_landmarks", None),
        getattr(results, "right_hand_landmarks", None),
    ]
    hand_landmarks = [landmarks for landmarks in hand_landmarks if landmarks is not None]
    if face_landmarks is None or not hand_landmarks:
        return _handle_hand_mouth_gap(config, now_ts)

    mouth_center = _mean_landmark_xy(
        face_landmarks,
        _FACE_LIP_CENTER_INDICES,
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    mouth_left = _safe_landmark_xy(
        face_landmarks,
        _FACE_LIP_CORNER_INDICES[0],
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    mouth_right = _safe_landmark_xy(
        face_landmarks,
        _FACE_LIP_CORNER_INDICES[1],
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    face_left = _safe_landmark_xy(
        face_landmarks,
        _FACE_WIDTH_INDICES[0],
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    face_right = _safe_landmark_xy(
        face_landmarks,
        _FACE_WIDTH_INDICES[1],
        crop_left,
        crop_top,
        input_w,
        input_h,
        pad_x,
        pad_y,
    )
    if mouth_center is None or mouth_left is None or mouth_right is None:
        return _handle_hand_mouth_gap(config, now_ts)

    mouth_width = _distance(mouth_left, mouth_right)
    face_width = 0.0
    if face_left is not None and face_right is not None:
        face_width = _distance(face_left, face_right)
    face_width = max(face_width, mouth_width * 2.8)
    if face_width < HAND_MOUTH_MIN_FACE_WIDTH_PX:
        return _handle_hand_mouth_gap(config, now_ts)

    closest_finger_xy = None
    closest_distance_ratio = float("inf")
    for landmarks in hand_landmarks:
        finger_xy = _safe_landmark_xy(
            landmarks,
            _INDEX_FINGER_TIP_INDEX,
            crop_left,
            crop_top,
            input_w,
            input_h,
            pad_x,
            pad_y,
        )
        if finger_xy is None:
            continue
        distance_ratio = _distance(finger_xy, mouth_center) / max(1.0, face_width)
        if distance_ratio < closest_distance_ratio:
            closest_distance_ratio = distance_ratio
            closest_finger_xy = finger_xy
    if closest_finger_xy is None:
        return _handle_hand_mouth_gap(config, now_ts)

    previous_distance_ratio = float(getattr(config, "person_proxy_last_distance_ratio", float("inf")))
    previous_finger_xy = getattr(config, "person_proxy_last_finger_xy", None)
    previous_mouth_xy = getattr(config, "person_proxy_last_mouth_xy", None)
    approach_delta_ratio = 0.0
    direction_cosine = 0.0
    if math.isfinite(previous_distance_ratio):
        approach_delta_ratio = max(0.0, previous_distance_ratio - closest_distance_ratio)
    if previous_finger_xy is not None:
        motion_vector = (
            closest_finger_xy[0] - previous_finger_xy[0],
            closest_finger_xy[1] - previous_finger_xy[1],
        )
        target_origin = previous_mouth_xy if previous_mouth_xy is not None else mouth_center
        target_vector = (
            target_origin[0] - previous_finger_xy[0],
            target_origin[1] - previous_finger_xy[1],
        )
        direction_cosine = _direction_cosine(motion_vector, target_vector)

    if (
        closest_distance_ratio <= (HAND_MOUTH_MAX_DISTANCE_RATIO * 1.35)
        and approach_delta_ratio >= HAND_MOUTH_MIN_APPROACH_DELTA_RATIO
        and direction_cosine >= HAND_MOUTH_MIN_DIRECTION_COSINE
    ):
        config.person_proxy_last_approach_ts = now_ts

    recent_approach = (
        now_ts - float(getattr(config, "person_proxy_last_approach_ts", 0.0))
    ) <= HAND_MOUTH_APPROACH_WINDOW_SECONDS
    within_threshold = closest_distance_ratio <= HAND_MOUTH_MAX_DISTANCE_RATIO

    dwell_started_at = float(getattr(config, "person_proxy_dwell_started_at", 0.0))
    if within_threshold:
        if recent_approach and dwell_started_at <= 0.0:
            config.person_proxy_dwell_started_at = now_ts
    else:
        config.person_proxy_dwell_started_at = 0.0

    dwell_started_at = float(getattr(config, "person_proxy_dwell_started_at", 0.0))
    dwell_elapsed = max(0.0, now_ts - dwell_started_at) if dwell_started_at > 0.0 else 0.0
    triggered = within_threshold and dwell_started_at > 0.0 and dwell_elapsed >= HAND_MOUTH_MIN_DWELL_SECONDS
    if triggered:
        config.person_proxy_active_until = now_ts + HAND_MOUTH_HOLD_SECONDS
    active_until = float(getattr(config, "person_proxy_active_until", 0.0))
    active = active_until > 0.0 and now_ts <= active_until

    config.person_proxy_last_seen_ts = now_ts
    config.person_proxy_last_distance_ratio = closest_distance_ratio
    config.person_proxy_last_finger_xy = closest_finger_xy
    config.person_proxy_last_mouth_xy = mouth_center

    score = _hand_mouth_score(
        distance_ratio=closest_distance_ratio,
        dwell_elapsed=dwell_elapsed,
        approach_delta_ratio=approach_delta_ratio,
        direction_cosine=direction_cosine,
        active=active,
    )
    return active, score

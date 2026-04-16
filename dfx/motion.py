"""Heuristic motion scoring for eating/drinking detection."""

import math
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

from dfx.constants import (
    CONSUMPTION_CLASS_NAMES,
    DRINK_CONTAINER_CLASS_NAMES,
    HANDHELD_FOOD_CLASS_NAMES,
    MOTION_TRIGGER_SCORE,
    PERSON_PROXY_CONFIRM_FRAMES,
    PERSON_PROXY_DIFF_THRESHOLD,
    PERSON_PROXY_HOLD_SECONDS,
    PERSON_PROXY_MIN_APPROACH_RATIO,
    PERSON_PROXY_MIN_AREA_RATIO,
    PERSON_PROXY_MIN_CONFIDENCE,
    PERSON_PROXY_MIN_MOUTH_RATIO,
    PERSON_PROXY_APPROACH_MOTION_RATIO,
    PERSON_PROXY_MOUTH_MOTION_RATIO,
    PERSON_PROXY_SCORE_FLOOR,
    PERSON_PROXY_TRIGGER_SCORE,
)


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
    """Fallback hand-to-mouth motion signal from person-only upper-face ROI movement."""
    if cv2 is None or np is None:
        return False, 0.0
    if not person_detections:
        history = getattr(config, "person_proxy_score_history", None)
        if history is not None:
            history.append(0.0)
        config.person_proxy_trigger_streak = 0
        return (now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))), 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    frame_area = max(1.0, float(frame_h * frame_w))
    downsample = 0.5
    small = cv2.resize(
        gray,
        (max(1, int(frame_w * downsample)), max(1, int(frame_h * downsample))),
        interpolation=cv2.INTER_AREA,
    )
    prev_small = getattr(config, "person_proxy_prev_gray", None)
    config.person_proxy_prev_gray = small
    if prev_small is None or getattr(prev_small, "shape", None) != small.shape:
        config.person_proxy_trigger_streak = 0
        return (now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))), 0.0

    diff = cv2.absdiff(small, prev_small)
    _, motion_mask = cv2.threshold(diff, PERSON_PROXY_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    best_mouth_ratio = 0.0
    best_approach_ratio = 0.0
    best_raw_score = 0.0
    scale_x = float(small.shape[1]) / max(1.0, float(frame_w))
    scale_y = float(small.shape[0]) / max(1.0, float(frame_h))
    for det in person_detections:
        try:
            person_confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            person_confidence = 0.0
        if person_confidence < PERSON_PROXY_MIN_CONFIDENCE:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        person_w = max(1.0, x2 - x1)
        person_h = max(1.0, y2 - y1)
        if (person_w * person_h) / frame_area < PERSON_PROXY_MIN_AREA_RATIO:
            continue

        # Focus on mouth/hand interaction area in upper-middle of the person box.
        roi_x1 = x1 + (person_w * 0.24)
        roi_x2 = x1 + (person_w * 0.76)
        roi_y1 = y1 + (person_h * 0.08)
        roi_y2 = y1 + (person_h * 0.48)

        sx1 = max(0, min(motion_mask.shape[1] - 1, int(roi_x1 * scale_x)))
        sy1 = max(0, min(motion_mask.shape[0] - 1, int(roi_y1 * scale_y)))
        sx2 = max(sx1 + 1, min(motion_mask.shape[1], int(roi_x2 * scale_x)))
        sy2 = max(sy1 + 1, min(motion_mask.shape[0], int(roi_y2 * scale_y)))
        roi = motion_mask[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            continue
        mouth_ratio = float(cv2.countNonZero(roi)) / float(roi.size)

        approach_x1 = x1 + (person_w * 0.18)
        approach_x2 = x1 + (person_w * 0.82)
        approach_y1 = y1 + (person_h * 0.22)
        approach_y2 = y1 + (person_h * 0.78)
        ax1 = max(0, min(motion_mask.shape[1] - 1, int(approach_x1 * scale_x)))
        ay1 = max(0, min(motion_mask.shape[0] - 1, int(approach_y1 * scale_y)))
        ax2 = max(ax1 + 1, min(motion_mask.shape[1], int(approach_x2 * scale_x)))
        ay2 = max(ay1 + 1, min(motion_mask.shape[0], int(approach_y2 * scale_y)))
        approach_roi = motion_mask[ay1:ay2, ax1:ax2]
        approach_ratio = 0.0
        if approach_roi.size > 0:
            approach_ratio = float(cv2.countNonZero(approach_roi)) / float(approach_roi.size)

        mouth_score = mouth_ratio / max(1e-6, PERSON_PROXY_MOUTH_MOTION_RATIO)
        approach_score = approach_ratio / max(1e-6, PERSON_PROXY_APPROACH_MOTION_RATIO)
        raw_score = (0.72 * mouth_score) + (0.28 * approach_score)
        if raw_score > best_raw_score:
            best_raw_score = raw_score
            best_mouth_ratio = mouth_ratio
            best_approach_ratio = approach_ratio

    score_history = getattr(config, "person_proxy_score_history", None)
    if score_history is not None:
        score_history.append(best_raw_score)
        smoothed_score = sum(score_history) / max(1, len(score_history))
    else:
        smoothed_score = best_raw_score

    candidate_trigger = (
        smoothed_score >= PERSON_PROXY_TRIGGER_SCORE
        and best_mouth_ratio >= PERSON_PROXY_MIN_MOUTH_RATIO
        and best_approach_ratio >= PERSON_PROXY_MIN_APPROACH_RATIO
    )
    if candidate_trigger:
        config.person_proxy_trigger_streak = int(getattr(config, "person_proxy_trigger_streak", 0)) + 1
    else:
        config.person_proxy_trigger_streak = max(0, int(getattr(config, "person_proxy_trigger_streak", 0)) - 1)
    triggered = int(getattr(config, "person_proxy_trigger_streak", 0)) >= PERSON_PROXY_CONFIRM_FRAMES
    if triggered:
        config.person_proxy_active_until = now_ts + PERSON_PROXY_HOLD_SECONDS
    active = now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))
    score = min(1.5, max(0.0, smoothed_score))
    if active:
        score = max(score, PERSON_PROXY_SCORE_FLOOR)
    return active, round(score, 3)

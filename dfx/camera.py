"""Camera capture, device management, and the main camera worker loop."""

import glob
import json
import math
import platform
import queue
import subprocess
import time
from collections import deque
from copy import deepcopy

try:
    import cv2
except Exception:
    cv2 = None

if cv2 is not None:
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

from dfx.constants import (
    ALERT_DETECTION_CONFIDENCE_FLOOR,
    ALERT_SNIPPET_CONFIDENCE_FLOOR,
    FOOD_CLASS_NAMES,
    FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE,
    HAND_TO_MOUTH_FOOD_VISIBILITY_FLOOR,
    FOOD_MOTION_CONFIRM_FRAMES,
    FOOD_MOTION_MIN_SCORE,
    FOOD_OCCLUSION_LOOKBACK_SECONDS,
    HAND_TO_MOUTH_REQUIRED_EVENTS,
    HAND_TO_MOUTH_WINDOW_SECONDS,
    INFERENCE_CLASS_NAMES,
    MOTION_TRIGGER_SCORE,
    NEW_OBJECT_MIN_ALERT_GAP_SECONDS,
    OCCLUDED_MOTION_HOLD_SECONDS,
    OCCLUDED_MOTION_PROXY_SCORE,
    PROXY_HAND_TO_MOUTH_REQUIRED_EVENTS,
    PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE,
    SAME_PERSON_ALERT_WINDOW_SECONDS,
    SAME_PERSON_MAX_ALERTS_IN_WINDOW,
    SAME_PERSON_SUPPRESSION_DISTANCE_RATIO,
    STATIONARY_FOLLOWUP_SECONDS,
)
from dfx.alerts import (
    append_alert,
    create_alert,
    has_novel_alert_object,
    remember_alert_objects,
    select_alert_person_center,
)
from dfx.detection import (
    detections_from_result,
    draw_detections,
    get_allowed_class_ids,
    make_status_frame,
)
from dfx.gpu import predict_with_fallback
from dfx.motion import (
    detect_consumption_motion,
    detect_person_hand_to_mouth_proxy,
    reset_person_hand_to_mouth_state,
)
from dfx.training import runtime_food_class_names, refresh_runtime_class_names


_ALERT_MIN_BOX_AREA_RATIO = 0.0012
_DRINK_ALERT_CLASS_NAMES = {"bottle", "cup", "drink", "drinks"}
_DRINK_ALERT_CONFIDENCE_FLOOR = max(0.70, ALERT_DETECTION_CONFIDENCE_FLOOR + 0.08)


def _camera_backend_flag() -> int:
    """Prefer AVFoundation on macOS so camera probing stays on the native backend."""
    if cv2 is None:
        return 0
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return int(cv2.CAP_AVFOUNDATION)
    return int(getattr(cv2, "CAP_ANY", 0))


def _open_camera_capture(index: int):
    """Open one camera index using the best backend for the current platform."""
    if cv2 is None:
        return None
    backend = _camera_backend_flag()
    capture = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    if capture is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _suggest_camera_probe_count(default_count: int) -> int:
    """Use the host OS to avoid probing obviously invalid camera indices."""
    if platform.system() == "Linux":
        try:
            discovered = len(glob.glob("/dev/video*"))
            if discovered > 0:
                return max(1, min(default_count, discovered))
        except OSError:
            pass
    if platform.system() != "Darwin":
        return max(1, default_count)
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
        cameras = payload.get("SPCameraDataType", [])
        if isinstance(cameras, list) and cameras:
            return max(1, min(default_count, len(cameras)))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return max(1, default_count)


def list_camera_devices(max_devices: int = 8) -> list[dict]:
    """Probe a small range of camera indices for use in the dashboard dropdown."""
    devices: list[dict] = []
    if cv2 is None:
        return devices
    probe_count = _suggest_camera_probe_count(max_devices)
    for index in range(probe_count):
        cap = _open_camera_capture(index)
        if cap is None:
            continue
        available = bool(cap.isOpened())
        if available:
            cap.release()
        devices.append(
            {
                "index": index,
                "label": f"Camera {index}",
                "available": available,
            }
        )
    return [device for device in devices if device["available"]] or devices[:1]


def _publish_frame(config, frame, jpeg_quality: int):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    jpeg_payload = encoded.tobytes() if ok else None
    frame_packet = (frame, jpeg_payload, time.time())

    frame_queue = getattr(config, "frame_queue", None)
    if frame_queue is not None:
        try:
            frame_queue.put_nowait(frame_packet)
        except queue.Full:
            # Keep only fresh frames under load by dropping one stale queued packet.
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(frame_packet)
            except queue.Full:
                pass
        return

    # Compatibility fallback if queue-based handoff is unavailable.
    with config.frame_lock:
        config.latest_frame = frame
        config.latest_jpeg = jpeg_payload


def _publish_status_frame(config, width: int, height: int, text: str, jpeg_quality: int):
    """Render and publish a simple status frame (camera off/unavailable)."""
    status = make_status_frame(width or 640, height or 360, text)
    if status is not None:
        _publish_frame(config, status, jpeg_quality)


def _reset_alert_debounce_state(config):
    """Reset debounce/arming counters used by initial and follow-up alerts."""
    config.consecutive = 0
    config.clear_count = 0
    config.armed = True
    config.stationary_first_alert_ts = 0.0
    config.stationary_followup_sent = False


def _reset_motion_and_person_state(config):
    """Reset motion/person-derived state when camera or detection is disabled."""
    config.motion_event_times.clear()
    config.last_motion_active = False
    config.last_food_seen_ts = 0.0
    config.occlusion_motion_until = 0.0
    reset_person_hand_to_mouth_state(config)
    config.food_motion_confirm_streak = 0
    config.person_alert_history.clear()
    config.alert_object_history.clear()
    config.motion_tracks.clear()
    config.next_motion_track_id = 1


def _is_same_person_suppressed(config, alert_person_center, wall_now: float, frame_diag: float) -> bool:
    """Return True when a recent alert likely belongs to the same person."""
    if alert_person_center is None:
        return False
    # Keep only recent person-alert entries in the configured time window.
    while (
        config.person_alert_history
        and (wall_now - float(config.person_alert_history[0][2])) > SAME_PERSON_ALERT_WINDOW_SECONDS
    ):
        config.person_alert_history.popleft()
    suppression_distance = SAME_PERSON_SUPPRESSION_DISTANCE_RATIO * max(1.0, frame_diag)
    matched_alerts = 0
    for px, py, pts in config.person_alert_history:
        if (wall_now - float(pts)) > SAME_PERSON_ALERT_WINDOW_SECONDS:
            continue
        distance = math.hypot(alert_person_center[0] - px, alert_person_center[1] - py)
        if distance <= suppression_distance:
            matched_alerts += 1
    return matched_alerts >= SAME_PERSON_MAX_ALERTS_IN_WINDOW


def _clamp_bbox_xyxy(bounds, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Clamp one bbox to valid frame coordinates."""
    x1, y1, x2, y2 = bounds
    left = max(0, min(frame_width - 1, int(round(float(x1)))))
    top = max(0, min(frame_height - 1, int(round(float(y1)))))
    right = max(left + 1, min(frame_width, int(round(float(x2)))))
    bottom = max(top + 1, min(frame_height, int(round(float(y2)))))
    return left, top, right, bottom


def _build_hand_to_mouth_alert_marker(config, frame, person_detections: list[dict], motion_score: float, motion_source: str):
    """Build a fallback boxed detection so proxy-only motion alerts always attach a snippet image."""
    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 1 or frame_h <= 1:
        return None

    marker_bbox = None
    best_confidence = -1.0
    for det in person_detections:
        bbox = det.get("bbox_xyxy") if isinstance(det, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            confidence = float(det.get("confidence", 0.0))
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if confidence > best_confidence:
            best_confidence = confidence
            marker_bbox = (x1, y1, x2, y2)

    mouth_xy = getattr(config, "person_proxy_last_mouth_xy", None)
    finger_xy = getattr(config, "person_proxy_last_finger_xy", None)
    if marker_bbox is None and mouth_xy is not None and finger_xy is not None:
        min_x = min(float(mouth_xy[0]), float(finger_xy[0]))
        min_y = min(float(mouth_xy[1]), float(finger_xy[1]))
        max_x = max(float(mouth_xy[0]), float(finger_xy[0]))
        max_y = max(float(mouth_xy[1]), float(finger_xy[1]))
        span = max(max_x - min_x, max_y - min_y, 40.0)
        pad = max(26.0, span * 0.80)
        marker_bbox = (min_x - pad, min_y - pad, max_x + pad, max_y + pad)

    if marker_bbox is None and (mouth_xy is not None or finger_xy is not None):
        anchor = mouth_xy if mouth_xy is not None else finger_xy
        anchor_x = float(anchor[0])
        anchor_y = float(anchor[1])
        pad = max(48.0, min(frame_w, frame_h) * 0.08)
        marker_bbox = (anchor_x - pad, anchor_y - pad, anchor_x + pad, anchor_y + pad)

    if marker_bbox is None:
        box_w = max(80.0, float(frame_w) * 0.35)
        box_h = max(80.0, float(frame_h) * 0.45)
        center_x = float(frame_w) * 0.5
        center_y = float(frame_h) * 0.45
        marker_bbox = (
            center_x - (box_w * 0.5),
            center_y - (box_h * 0.5),
            center_x + (box_w * 0.5),
            center_y + (box_h * 0.5),
        )

    left, top, right, bottom = _clamp_bbox_xyxy(marker_bbox, frame_w, frame_h)
    center_x = round((left + right) * 0.5, 2)
    center_y = round((top + bottom) * 0.5, 2)
    return {
        "class_id": -1,
        "class_name": "hand_to_mouth",
        "confidence": round(max(ALERT_SNIPPET_CONFIDENCE_FLOOR + 0.05, float(motion_score)), 4),
        "bbox_xyxy": [left, top, right, bottom],
        "center_xy": [center_x, center_y],
        "hand_to_mouth_source": str(motion_source or "person_proxy"),
    }


def _person_centers(person_detections: list[dict]) -> list[tuple[float, float]]:
    """Return person centers used to bias alerts toward likely real interactions."""
    centers: list[tuple[float, float]] = []
    for det in person_detections:
        center = det.get("center_xy") if isinstance(det, dict) else None
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        try:
            centers.append((float(center[0]), float(center[1])))
        except (TypeError, ValueError):
            continue
    return centers


def _detection_area_ratio(det: dict, frame_w: int, frame_h: int) -> float:
    """Compute one detection area ratio relative to the current frame."""
    bbox = det.get("bbox_xyxy") if isinstance(det, dict) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return 0.0
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_w) * float(frame_h))
    return (width * height) / frame_area


def _detection_min_person_distance(det: dict, people_centers: list[tuple[float, float]]) -> float | None:
    """Return nearest-person distance for one detection center, or None if unavailable."""
    if not people_centers:
        return None
    center = det.get("center_xy") if isinstance(det, dict) else None
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    try:
        cx = float(center[0])
        cy = float(center[1])
    except (TypeError, ValueError):
        return None
    return min(math.hypot(cx - px, cy - py) for px, py in people_centers)


def _is_plausible_alert_detection(det: dict, frame_w: int, frame_h: int) -> bool:
    """Filter noisy detections (tiny specks or weak drink-container guesses)."""
    try:
        confidence = float(det.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    class_name = str(det.get("class_name", "")).strip().lower()
    required_confidence = ALERT_DETECTION_CONFIDENCE_FLOOR
    if class_name in _DRINK_ALERT_CLASS_NAMES:
        required_confidence = max(required_confidence, _DRINK_ALERT_CONFIDENCE_FLOOR)
    if confidence < required_confidence:
        return False
    return _detection_area_ratio(det, frame_w, frame_h) >= _ALERT_MIN_BOX_AREA_RATIO


def _select_primary_alert_detections(
    alert_detections: list[dict],
    person_detections: list[dict],
    frame_shape,
) -> list[dict]:
    """Pick one primary detection per alert so accept/reject actions stay atomic."""
    if not alert_detections:
        return []
    frame_h = int(frame_shape[0]) if len(frame_shape) >= 1 else 0
    frame_w = int(frame_shape[1]) if len(frame_shape) >= 2 else 0
    if frame_w <= 1 or frame_h <= 1:
        return [alert_detections[0]]

    people_centers = _person_centers(person_detections)
    plausible = [
        det for det in alert_detections if _is_plausible_alert_detection(det, frame_w, frame_h)
    ]
    candidates = plausible or list(alert_detections)
    frame_diag = max(1.0, math.hypot(float(frame_w), float(frame_h)))

    def _score(det: dict) -> float:
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        area_bonus = min(0.2, _detection_area_ratio(det, frame_w, frame_h) * 20.0)
        nearest_person = _detection_min_person_distance(det, people_centers)
        if nearest_person is None:
            proximity_bonus = 0.0
        else:
            proximity_bonus = max(0.0, 0.18 - ((nearest_person / frame_diag) * 0.5))
        return confidence + area_bonus + proximity_bonus

    best = max(candidates, key=_score)
    return [best]


def _persist_alert_synchronously(config, payload: dict):
    """Compatibility path: persist one alert inline when no alert queue is configured."""
    alert = create_alert(
        payload["frame"],
        payload["detections"],
        snippet_dir=payload["snippet_dir"],
        video_dir=payload["video_dir"],
        recent_frames=payload["recent_frames"],
        video_fps=payload["video_fps"],
        camera_zone=payload["camera_zone"],
        context_detections=payload["context_detections"],
        motion_detected=payload["motion_detected"],
        motion_score=payload["motion_score"],
        hand_to_mouth_source=payload["hand_to_mouth_source"],
        hand_to_mouth_event_count=payload["hand_to_mouth_event_count"],
        attach_video=payload["attach_video"],
        alert_reason=payload["alert_reason"],
    )
    with config.alert_lock:
        append_alert(
            config.alert_log,
            alert,
            summary_csv_path=config.detection_summary_csv,
        )


def _enqueue_alert_job(config, payload: dict) -> bool:
    """Queue one alert job for background persistence; drops stale jobs under sustained load."""
    alert_queue = getattr(config, "alert_queue", None)
    if alert_queue is None:
        return False
    try:
        alert_queue.put_nowait(payload)
        return True
    except queue.Full:
        # Preserve recent alerts by evicting one stale queued item.
        try:
            alert_queue.get_nowait()
            alert_queue.task_done()
        except queue.Empty:
            pass
        try:
            alert_queue.put_nowait(payload)
            config.alert_jobs_dropped = int(getattr(config, "alert_jobs_dropped", 0)) + 1
            return True
        except queue.Full:
            config.alert_jobs_dropped = int(getattr(config, "alert_jobs_dropped", 0)) + 1
            return False


def camera_worker(config, cam_index: int):
    """Capture frames, run detection, update the stream frame, and create alerts."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for live camera mode.")
    cap = None
    active_cam_index = int(cam_index)
    last_detections = []
    last_motion_detected = False
    last_motion_score = 0.0
    next_inference_at = 0.0
    allowed_ids = None
    active_model = None
    active_allowed_names: frozenset[str] = frozenset()
    recent_frames: deque = deque(maxlen=80)

    try:
        while not config.stop:
            loop_started_at = time.perf_counter()
            # Snapshot the tunable settings once per loop so the frame is processed consistently.
            with config.settings_lock:
                camera_enabled = bool(config.camera_enabled)
                detection_enabled = bool(config.detection_enabled)
                motion_enabled = bool(config.motion_enabled)
                conf = float(config.conf)
                iou = float(config.iou)
                persist_frames = int(config.persist_frames)
                cooldown = float(config.cooldown)
                clear_frames = int(config.clear_frames)
                stream_fps = float(config.stream_fps)
                out_width = int(config.width)
                out_height = int(config.height)
                inference_imgsz = int(config.inference_imgsz)
                max_inference_fps = float(config.max_inference_fps)
                jpeg_quality = int(config.jpeg_quality)
                camera_index = int(config.camera_index)
                camera_zone = str(config.camera_zone)

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                    active_cam_index = camera_index
                # Turning the camera off also resets the alert state machine.
                _reset_alert_debounce_state(config)
                _reset_motion_and_person_state(config)
                last_detections = []
                last_motion_detected = False
                last_motion_score = 0.0
                recent_frames.clear()
                _publish_status_frame(config, out_width, out_height, "Camera is OFF", jpeg_quality)
                time.sleep(0.15)
                continue

            if cap is not None and active_cam_index != camera_index:
                cap.release()
                cap = None
                active_cam_index = camera_index
                allowed_ids = None
                next_inference_at = 0.0

            if cap is None:
                cap = _open_camera_capture(camera_index)
                if cap is None:
                    raise RuntimeError("OpenCV camera backend is not available.")
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    _publish_status_frame(
                        config,
                        out_width,
                        out_height,
                        f"Camera {camera_index} unavailable",
                        jpeg_quality,
                    )
                    time.sleep(1.0)
                    continue
                active_cam_index = camera_index
                allowed_ids = None
                next_inference_at = 0.0

            wall_now = time.time()

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                allowed_ids = None
                recent_frames.clear()
                time.sleep(0.1)
                continue

            # Cameras are physically mounted upside down; rotate before detection and streaming.
            frame = cv2.flip(frame, -1)

            recent_frames.append(frame.copy())

            detections = last_detections
            motion_detected = last_motion_detected
            motion_score = last_motion_score
            motion_source = "none"
            inference_ran = False
            hand_to_mouth_event_active = False
            all_detections = detections
            alert_detections: list[dict] = []
            primary_alert_detections: list[dict] = []
            visible_food_detections: list[dict] = []
            person_detections: list[dict] = []
            if detection_enabled:
                perf_now = time.perf_counter()
                inference_due = max_inference_fps <= 0.0 or perf_now >= next_inference_at
                if inference_due:
                    inference_ran = True
                    if max_inference_fps > 0.0:
                        next_inference_at = perf_now + (1.0 / max(0.1, max_inference_fps))
                    else:
                        next_inference_at = 0.0
                with config.model_lock:
                    model = config.model
                    if inference_ran:
                        allowed_names = set(
                            getattr(config, "runtime_inference_class_names", INFERENCE_CLASS_NAMES)
                        )
                        tracked_names = set(
                            getattr(config, "runtime_food_class_names", runtime_food_class_names(config))
                        )
                        allowed_names_key = frozenset(allowed_names)
                        if model is not active_model or allowed_names_key != active_allowed_names:
                            active_model = model
                            active_allowed_names = allowed_names_key
                            allowed_ids = None
                        if allowed_ids is None:
                            allowed_ids = get_allowed_class_ids(model, allowed_names)
                        predict_kwargs = {
                            "verbose": False,
                            "conf": conf,
                            "iou": iou,
                            "imgsz": inference_imgsz,
                            "classes": allowed_ids if allowed_ids else None,
                            "device": getattr(config, "inference_device", "cpu"),
                        }
                        results = predict_with_fallback(model, frame, **predict_kwargs)
                        selected_device = str(
                            getattr(model, "_dfx_inference_device_override", predict_kwargs["device"])
                        ).strip() or "cpu"
                        if selected_device != str(getattr(config, "inference_device", "cpu")):
                            config.inference_device = selected_device
                            print(f"Warning: switched inference device to {selected_device}")
                if inference_ran:
                    result = results[0]
                    all_detections = detections_from_result(result, allowed_names=allowed_names)
                    detections = [
                        det
                        for det in all_detections
                        if str(det.get("class_name", "")).strip().lower() in tracked_names
                    ]
                    alert_detections = [
                        det
                        for det in detections
                        if (
                            float(det.get("confidence", 0.0)) >= ALERT_DETECTION_CONFIDENCE_FLOOR
                        )
                    ]
                    visible_food_detections = [
                        det
                        for det in detections
                        if float(det.get("confidence", 0.0)) >= HAND_TO_MOUTH_FOOD_VISIBILITY_FLOOR
                    ]
                    person_detections = [
                        det
                        for det in all_detections
                        if str(det.get("class_name", "")).strip().lower() == "person"
                    ]
                    primary_alert_detections = _select_primary_alert_detections(
                        alert_detections,
                        person_detections,
                        frame.shape,
                    )
                    if motion_enabled:
                        raw_motion_detected, raw_motion_score = detect_consumption_motion(
                            config,
                            detections,
                            frame_width=int(frame.shape[1]),
                            frame_height=int(frame.shape[0]),
                            person_detections=person_detections,
                        )
                        if raw_motion_detected and raw_motion_score >= FOOD_MOTION_MIN_SCORE:
                            config.food_motion_confirm_streak = min(
                                FOOD_MOTION_CONFIRM_FRAMES + 2,
                                int(getattr(config, "food_motion_confirm_streak", 0)) + 1,
                            )
                        else:
                            config.food_motion_confirm_streak = 0
                        motion_detected = int(getattr(config, "food_motion_confirm_streak", 0)) >= FOOD_MOTION_CONFIRM_FRAMES
                        motion_score = float(raw_motion_score)
                        if motion_detected:
                            motion_source = "food_track"
                        person_proxy_detected, person_proxy_score = detect_person_hand_to_mouth_proxy(
                            config,
                            frame,
                            person_detections,
                            wall_now,
                        )
                        if person_proxy_detected and (
                            not motion_detected or float(person_proxy_score) >= float(motion_score)
                        ):
                            # Landmark proximity can confirm eating/drinking even when object-track motion is weak.
                            motion_detected = True
                            motion_score = max(float(motion_score), float(person_proxy_score))
                            motion_source = "person_proxy"
                        if visible_food_detections:
                            config.last_food_seen_ts = wall_now
                        if not motion_detected:
                            # Keep motion active briefly when food is momentarily occluded by a hand.
                            recently_saw_food = (
                                (wall_now - float(config.last_food_seen_ts)) <= FOOD_OCCLUSION_LOOKBACK_SECONDS
                            )
                            if (
                                not visible_food_detections
                                and bool(person_detections)
                                and recently_saw_food
                                and wall_now <= float(config.occlusion_motion_until)
                            ):
                                motion_detected = True
                                motion_score = max(float(motion_score), OCCLUDED_MOTION_PROXY_SCORE)
                                motion_source = "food_occluded"
                        if motion_detected:
                            config.occlusion_motion_until = wall_now + OCCLUDED_MOTION_HOLD_SECONDS
                    else:
                        config.motion_tracks.clear()
                        config.next_motion_track_id = 1
                        config.food_motion_confirm_streak = 0
                        motion_detected = False
                        motion_score = 0.0

                    hand_to_mouth_event_active = (
                        motion_detected
                        and (
                            (
                                motion_source == "food_track"
                                and bool(visible_food_detections)
                                and float(motion_score) >= FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE
                            )
                            or (
                                motion_source in {"person_proxy", "food_occluded"}
                                and float(motion_score) >= PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE
                            )
                        )
                    )

                    if hand_to_mouth_event_active and not config.last_motion_active:
                        config.motion_event_times.append(wall_now)
                    while (
                        config.motion_event_times
                        and (wall_now - config.motion_event_times[0]) > HAND_TO_MOUTH_WINDOW_SECONDS
                    ):
                        config.motion_event_times.popleft()
                    config.last_motion_active = bool(hand_to_mouth_event_active)

                    last_detections = detections
                    last_motion_detected = motion_detected
                    last_motion_score = motion_score
            else:
                _reset_motion_and_person_state(config)
                last_detections = []
                last_motion_detected = False
                last_motion_score = 0.0
                next_inference_at = 0.0
                detections = []
                motion_detected = False
                motion_score = 0.0

            # This debounce logic makes "item stays in view" produce one alert rather than many.
            if detection_enabled and inference_ran and primary_alert_detections:
                config.consecutive += 1
                config.clear_count = 0
            elif detection_enabled and inference_ran:
                config.consecutive = 0
                config.clear_count += 1
                if config.clear_count >= max(1, clear_frames):
                    config.armed = True
                    config.stationary_first_alert_ts = 0.0
                    config.stationary_followup_sent = False
            elif not detection_enabled:
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True
                config.stationary_first_alert_ts = 0.0
                config.stationary_followup_sent = False

            motion_burst_trigger = (
                inference_ran
                and hand_to_mouth_event_active
                and len(config.motion_event_times) >= (
                    PROXY_HAND_TO_MOUTH_REQUIRED_EVENTS
                    if motion_source in {"person_proxy", "food_occluded"}
                    else HAND_TO_MOUTH_REQUIRED_EVENTS
                )
            )
            stationary_followup_trigger = (
                inference_ran
                and bool(primary_alert_detections)
                and not motion_detected
                and config.stationary_first_alert_ts > 0.0
                and not config.stationary_followup_sent
                and (wall_now - config.stationary_first_alert_ts) >= STATIONARY_FOLLOWUP_SECONDS
            )
            initial_trigger = (
                inference_ran
                and primary_alert_detections
                and config.consecutive >= max(1, persist_frames)
                and config.armed
                and (wall_now - config.last_alert_ts) >= max(0.0, cooldown)
            )
            frame_diag = math.hypot(float(frame.shape[1]), float(frame.shape[0]))
            new_object_trigger = (
                inference_ran
                and bool(primary_alert_detections)
                and (wall_now - config.last_alert_ts) >= NEW_OBJECT_MIN_ALERT_GAP_SECONDS
                and has_novel_alert_object(
                    config,
                    primary_alert_detections,
                    frame_diag=frame_diag,
                    now_ts=wall_now,
                )
            )

            same_person_suppressed = False
            alert_person_center = select_alert_person_center(person_detections, primary_alert_detections)
            same_person_suppressed = _is_same_person_suppressed(
                config,
                alert_person_center,
                wall_now,
                frame_diag,
            )

            should_alert = bool(motion_burst_trigger) or (
                not same_person_suppressed
                and (stationary_followup_trigger or initial_trigger or new_object_trigger)
            )
            if should_alert:
                reason = "initial"
                if motion_burst_trigger:
                    reason = "motion_burst"
                elif new_object_trigger:
                    reason = "new_object"
                elif stationary_followup_trigger:
                    reason = "stationary_followup"

                payload_detections = deepcopy(primary_alert_detections)
                if hand_to_mouth_event_active:
                    has_snippet_candidate = any(
                        float(det.get("confidence", 0.0)) >= ALERT_SNIPPET_CONFIDENCE_FLOOR
                        for det in payload_detections
                        if isinstance(det, dict)
                    )
                    if not has_snippet_candidate:
                        marker = _build_hand_to_mouth_alert_marker(
                            config,
                            frame,
                            person_detections,
                            motion_score,
                            motion_source,
                        )
                        if marker is not None:
                            payload_detections.append(marker)

                recent_frames_snapshot = list(recent_frames)
                if motion_burst_trigger and len(recent_frames_snapshot) < 3:
                    recent_frames_snapshot.extend([frame.copy() for _ in range(3 - len(recent_frames_snapshot))])

                alert_payload = {
                    "frame": frame.copy(),
                    "detections": payload_detections,
                    "snippet_dir": config.snippet_dir,
                    "video_dir": config.video_dir,
                    "recent_frames": recent_frames_snapshot,
                    "video_fps": stream_fps,
                    "camera_zone": camera_zone,
                    "context_detections": deepcopy(all_detections),
                    "motion_detected": motion_detected,
                    "motion_score": motion_score,
                    "hand_to_mouth_source": motion_source,
                    "hand_to_mouth_event_count": len(config.motion_event_times),
                    "attach_video": bool(motion_burst_trigger or hand_to_mouth_event_active),
                    "alert_reason": reason,
                }
                if not _enqueue_alert_job(config, alert_payload):
                    _persist_alert_synchronously(config, alert_payload)
                config.last_alert_ts = wall_now
                remember_alert_objects(config, primary_alert_detections, wall_now)
                if alert_person_center is not None:
                    config.person_alert_history.append(
                        (
                            float(alert_person_center[0]),
                            float(alert_person_center[1]),
                            float(wall_now),
                        )
                    )
                if motion_burst_trigger:
                    # Immediate burst alerts bypass cooldown/arming but should not spam every frame.
                    config.motion_event_times.clear()
                elif stationary_followup_trigger:
                    config.stationary_followup_sent = True
                elif new_object_trigger:
                    config.armed = True
                    config.stationary_first_alert_ts = 0.0
                    config.stationary_followup_sent = False
                else:
                    config.armed = False
                    if motion_detected:
                        config.stationary_first_alert_ts = 0.0
                        config.stationary_followup_sent = False
                    else:
                        config.stationary_first_alert_ts = wall_now
                        config.stationary_followup_sent = False

            annotated = draw_detections(frame, detections) if detection_enabled else frame.copy()
            if detection_enabled and motion_detected:
                cv2.putText(
                    annotated,
                    f"Eating/Drinking motion detected ({motion_score:.2f})",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (18, 200, 18),
                    2,
                )
            if not detection_enabled:
                cv2.putText(
                    annotated,
                    "Detection is OFF",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            if out_width or out_height:
                annotated = cv2.resize(
                    annotated,
                    (
                        out_width or annotated.shape[1],
                        out_height or annotated.shape[0],
                    ),
                )

            _publish_frame(config, annotated, jpeg_quality)

            target_loop_delay = 1.0 / max(1.0, stream_fps)
            remaining = target_loop_delay - (time.perf_counter() - loop_started_at)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if cap is not None:
            cap.release()

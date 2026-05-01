"""Camera capture, device management, and the main camera worker loop."""

import glob
import json
import logging
import math
import os
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
    FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE,
    HAND_MOUTH_SCORE_FLOOR,
    HAND_TO_MOUTH_FOOD_VISIBILITY_FLOOR,
    FOOD_MOTION_CONFIRM_FRAMES,
    FOOD_MOTION_MIN_SCORE,
    FOOD_OCCLUSION_LOOKBACK_SECONDS,
    HAND_TO_MOUTH_PERSON_COOLDOWN_SECONDS,
    HAND_TO_MOUTH_PERSON_TRACK_MATCH_DISTANCE_RATIO,
    HAND_TO_MOUTH_PERSON_TRACK_MAX_GAP_SECONDS,
    HAND_TO_MOUTH_REQUIRED_EVENTS,
    HAND_TO_MOUTH_VIDEO_BUFFER_SECONDS,
    HAND_TO_MOUTH_WINDOW_SECONDS,
    INFERENCE_CLASS_NAMES,
    NEW_OBJECT_MIN_ALERT_GAP_SECONDS,
    OCCLUDED_MOTION_HOLD_SECONDS,
    OCCLUDED_MOTION_PROXY_SCORE,
    PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE,
    SAME_PERSON_ALERT_WINDOW_SECONDS,
    SAME_PERSON_MAX_ALERTS_IN_WINDOW,
    SAME_PERSON_SUPPRESSION_DISTANCE_RATIO,
    STATIONARY_FOLLOWUP_SECONDS,
    is_ignored_yolo_class_name,
    normalize_class_label,
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
from dfx.training import runtime_food_class_names


logger = logging.getLogger(__name__)


_ALERT_MIN_BOX_AREA_RATIO = 0.0012
_DRINK_ALERT_CLASS_NAMES = {"bottle", "cup", "drink", "drinks"}
_DRINK_ALERT_CONFIDENCE_FLOOR = max(0.70, ALERT_DETECTION_CONFIDENCE_FLOOR + 0.08)
_PROXY_GESTURE_DEBOUNCE_SECONDS = 2.0
_PROXY_ALERT_SETTLE_SECONDS = 2.5
_PROXY_MIN_BUFFER_SECONDS = 2.0


def _camera_backend_flag() -> int:
    """Prefer AVFoundation on macOS so camera probing stays on the native backend."""
    if cv2 is None:
        return 0
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return int(cv2.CAP_AVFOUNDATION)
    if platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
        # Jetson and V4L2 USB cameras are generally more stable on the native V4L2 backend.
        return int(cv2.CAP_V4L2)
    return int(getattr(cv2, "CAP_ANY", 0))


def _is_jetson_linux_host() -> bool:
    """Return whether this process appears to be running on a Jetson Linux host."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    return "tegra" in platform.release().lower()


def _build_v4l2_gstreamer_pipelines(index: int, width: int, height: int, stream_fps: float) -> list[tuple[str, str]]:
    """Build Linux V4L2 GStreamer pipeline fallbacks for one camera index."""
    device_path = f"/dev/video{int(index)}"
    fps = max(5, min(60, int(round(float(stream_fps))) if stream_fps > 0 else 30))
    target_width = max(320, int(width) if int(width) > 0 else 1280)
    target_height = max(240, int(height) if int(height) > 0 else 720)

    pipelines: list[tuple[str, str]] = []
    if _is_jetson_linux_host():
        # Fast path for Jetson USB MJPEG cameras through NVIDIA accelerated decode.
        pipelines.append(
            (
                "gstreamer-jetson-mjpeg",
                " ! ".join(
                    [
                        f"v4l2src device={device_path} io-mode=2",
                        f"image/jpeg,width={target_width},height={target_height},framerate={fps}/1",
                        "jpegparse",
                        "nvv4l2decoder mjpeg=1",
                        "nvvidconv",
                        "video/x-raw,format=BGRx",
                        "videoconvert",
                        "video/x-raw,format=BGR",
                        "appsink drop=1 max-buffers=1 sync=false",
                    ]
                ),
            )
        )

    pipelines.append(
        (
            "gstreamer-v4l2-mjpeg",
            " ! ".join(
                [
                    f"v4l2src device={device_path} io-mode=2",
                    f"image/jpeg,width={target_width},height={target_height},framerate={fps}/1",
                    "jpegdec",
                    "videoconvert",
                    "video/x-raw,format=BGR",
                    "appsink drop=1 max-buffers=1 sync=false",
                ]
            ),
        )
    )
    pipelines.append(
        (
            "gstreamer-v4l2-raw",
            " ! ".join(
                [
                    f"v4l2src device={device_path} io-mode=2",
                    f"video/x-raw,width={target_width},height={target_height},framerate={fps}/1",
                    "videoconvert",
                    "video/x-raw,format=BGR",
                    "appsink drop=1 max-buffers=1 sync=false",
                ]
            ),
        )
    )
    return pipelines


def _warm_camera_stream(capture, warmup_reads: int = 4) -> bool:
    """Attempt a few initial reads so freshly-opened cameras can settle."""
    if capture is None:
        return False
    for _ in range(max(1, int(warmup_reads))):
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True
    return False


def _iter_model_names(model) -> list[tuple[int, str]]:
    """Return model class names as (id, name) pairs across YOLO wrappers."""
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return [(int(class_id), str(name)) for class_id, name in names.items()]
    if isinstance(names, list):
        return [(int(class_id), str(name)) for class_id, name in enumerate(names)]
    return []


def _ignored_yolo_class_ids(model) -> set[int]:
    """Resolve class IDs that should be stripped from raw YOLO inference output."""
    ignored_ids: set[int] = set()
    for class_id, class_name in _iter_model_names(model):
        if is_ignored_yolo_class_name(normalize_class_label(class_name)):
            ignored_ids.add(int(class_id))
    return ignored_ids


def _strip_result_boxes_by_class_ids(result, blocked_class_ids: set[int]) -> int:
    """Remove blocked-class boxes from one Ultralytics result object in place."""
    if not blocked_class_ids:
        return 0
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    cls_values = getattr(boxes, "cls", None)
    if cls_values is None:
        return 0
    try:
        total = int(len(cls_values))
    except Exception:
        return 0
    if total <= 0:
        return 0

    keep_indices: list[int] = []
    for idx in range(total):
        try:
            class_id = int(cls_values[idx])
        except Exception:
            keep_indices.append(idx)
            continue
        if class_id not in blocked_class_ids:
            keep_indices.append(idx)
    removed = total - len(keep_indices)
    if removed <= 0:
        return 0

    if hasattr(boxes, "__getitem__"):
        try:
            result.boxes = boxes[keep_indices]
            return removed
        except Exception:
            try:
                import torch

                device = getattr(cls_values, "device", None)
                keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
                result.boxes = boxes[keep_tensor]
                return removed
            except Exception:
                pass

    try:
        if isinstance(getattr(boxes, "xyxy", None), list):
            boxes.xyxy = [boxes.xyxy[idx] for idx in keep_indices]
        if isinstance(getattr(boxes, "conf", None), list):
            boxes.conf = [boxes.conf[idx] for idx in keep_indices]
        if isinstance(getattr(boxes, "cls", None), list):
            boxes.cls = [boxes.cls[idx] for idx in keep_indices]
        result.boxes = boxes
        return removed
    except Exception:
        return 0


def _open_camera_capture_with_fallback(
    index: int,
    *,
    width: int = 0,
    height: int = 0,
    stream_fps: float = 0.0,
) -> tuple[object | None, str, list[str]]:
    """Try multiple OpenCV/GStreamer strategies and return capture plus diagnostics."""
    if cv2 is None:
        return None, "OpenCV unavailable", ["OpenCV is unavailable"]

    diagnostics: list[str] = []
    attempts: list[tuple[str, object, int]] = []
    seen_attempts: set[tuple[str, int]] = set()

    def _append_attempt(label: str, source: object, backend: int):
        key = (str(source), int(backend))
        if key in seen_attempts:
            return
        seen_attempts.add(key)
        attempts.append((label, source, backend))

    if platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
        _append_attempt("opencv-v4l2-index", int(index), int(cv2.CAP_V4L2))
    preferred_backend = int(_camera_backend_flag())
    _append_attempt("opencv-preferred-index", int(index), preferred_backend)
    if hasattr(cv2, "CAP_ANY"):
        _append_attempt("opencv-cap-any-index", int(index), int(cv2.CAP_ANY))

    device_path = f"/dev/video{int(index)}"
    if (
        platform.system() == "Linux"
        and os.path.exists(device_path)
        and hasattr(cv2, "CAP_GSTREAMER")
    ):
        for label, pipeline in _build_v4l2_gstreamer_pipelines(index, width, height, stream_fps):
            _append_attempt(label, pipeline, int(cv2.CAP_GSTREAMER))

    for label, source, backend in attempts:
        capture = None
        try:
            capture = cv2.VideoCapture(source, backend) if int(backend) else cv2.VideoCapture(source)
        except Exception as exc:
            diagnostics.append(f"{label}: constructor error ({exc})")
            continue

        if capture is None:
            diagnostics.append(f"{label}: constructor returned None")
            continue

        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if int(width) > 0 and hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if int(height) > 0 and hasattr(cv2, "CAP_PROP_FRAME_HEIGHT"):
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if float(stream_fps) > 0 and hasattr(cv2, "CAP_PROP_FPS"):
                capture.set(cv2.CAP_PROP_FPS, float(stream_fps))
            if platform.system() == "Linux" and hasattr(cv2, "CAP_PROP_FOURCC"):
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            # Property setting is best-effort and backend-specific.
            pass

        if not capture.isOpened():
            diagnostics.append(f"{label}: open failed")
            capture.release()
            continue

        if not _warm_camera_stream(capture):
            diagnostics.append(f"{label}: opened but failed to read warmup frames")
            capture.release()
            continue

        actual_width = int(capture.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)) or 0)
        actual_height = int(capture.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)) or 0)
        actual_fps = float(capture.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0.0)
        diagnostics.append(
            f"{label}: opened ({actual_width}x{actual_height} @ {actual_fps:.2f} fps)"
        )
        return capture, label, diagnostics

    return None, f"Camera {int(index)} unavailable", diagnostics


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
    config.person_proxy_tracks.clear()
    config.next_person_proxy_track_id = 1
    config.person_proxy_event_state.clear()


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


def _extract_bbox_xyxy(det: dict) -> tuple[float, float, float, float] | None:
    """Return a normalized bbox tuple from one detection dict."""
    bbox = det.get("bbox_xyxy") if isinstance(det, dict) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_center_xy(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return one bbox center as (x, y)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Return IoU overlap for two bboxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter_area / max(1.0, (area_a + area_b - inter_area))


def _point_in_bbox(point_xy: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Return True when one point lies within one bbox."""
    px, py = point_xy
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _update_proxy_person_tracks(config, person_detections: list[dict], now_ts: float, frame_diag: float) -> None:
    """Attach stable local person IDs to current detections using IoU + center distance."""
    tracks = getattr(config, "person_proxy_tracks", None)
    if not isinstance(tracks, dict):
        tracks = {}
        config.person_proxy_tracks = tracks

    max_gap = max(0.5, float(HAND_TO_MOUTH_PERSON_TRACK_MAX_GAP_SECONDS))
    for track_id, track in list(tracks.items()):
        if (now_ts - float(track.get("last_seen_ts", 0.0))) > max_gap:
            tracks.pop(track_id, None)

    max_match_distance = max(
        40.0,
        float(HAND_TO_MOUTH_PERSON_TRACK_MATCH_DISTANCE_RATIO) * max(1.0, float(frame_diag)),
    )
    unmatched_track_ids: set[int] = set(tracks.keys())

    for det in sorted(
        person_detections,
        key=lambda entry: float(entry.get("confidence", 0.0)) if isinstance(entry, dict) else 0.0,
        reverse=True,
    ):
        bbox = _extract_bbox_xyxy(det)
        if bbox is None:
            continue
        center = _bbox_center_xy(bbox)

        best_track_id = None
        best_score = float("-inf")
        for track_id in list(unmatched_track_ids):
            track = tracks.get(track_id)
            if not isinstance(track, dict):
                continue
            track_bbox_raw = track.get("bbox_xyxy")
            if not isinstance(track_bbox_raw, (list, tuple)) or len(track_bbox_raw) != 4:
                continue
            try:
                track_bbox = tuple(float(v) for v in track_bbox_raw)
            except (TypeError, ValueError):
                continue
            track_center = _bbox_center_xy(track_bbox)
            center_distance = math.hypot(center[0] - track_center[0], center[1] - track_center[1])
            iou = _bbox_iou(bbox, track_bbox)
            if center_distance > max_match_distance and iou < 0.10:
                continue
            score = (iou * 2.0) - (center_distance / max(1.0, max_match_distance))
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = int(getattr(config, "next_person_proxy_track_id", 1))
            config.next_person_proxy_track_id = best_track_id + 1
            tracks[best_track_id] = {}

        tracks[best_track_id] = {
            "bbox_xyxy": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "center_xy": [float(center[0]), float(center[1])],
            "last_seen_ts": float(now_ts),
        }
        det["person_proxy_track_id"] = int(best_track_id)
        unmatched_track_ids.discard(best_track_id)


def _resolve_proxy_person_track_id(person_detections: list[dict], proxy_details: dict | None) -> int | None:
    """Resolve which person track owns the current proxy hand-to-mouth geometry."""
    if not isinstance(proxy_details, dict):
        return None

    target_bbox = proxy_details.get("subject_bbox_xyxy")
    bbox_tuple = None
    if isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4:
        try:
            bbox_tuple = tuple(float(v) for v in target_bbox)
        except (TypeError, ValueError):
            bbox_tuple = None

    target_point = None
    wrist_xy = proxy_details.get("wrist_xy")
    mouth_xy = proxy_details.get("mouth_xy")
    if isinstance(wrist_xy, (list, tuple)) and len(wrist_xy) == 2:
        try:
            target_point = (float(wrist_xy[0]), float(wrist_xy[1]))
        except (TypeError, ValueError):
            target_point = None
    if target_point is None and isinstance(mouth_xy, (list, tuple)) and len(mouth_xy) == 2:
        try:
            target_point = (float(mouth_xy[0]), float(mouth_xy[1]))
        except (TypeError, ValueError):
            target_point = None

    best_track_id = None
    best_score = float("-inf")
    for det in person_detections:
        track_id = det.get("person_proxy_track_id") if isinstance(det, dict) else None
        if track_id is None:
            continue
        bbox = _extract_bbox_xyxy(det)
        if bbox is None:
            continue

        score = 0.0
        if bbox_tuple is not None:
            score += _bbox_iou(bbox_tuple, bbox) * 2.0
        if target_point is not None:
            center = _bbox_center_xy(bbox)
            box_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            center_distance = math.hypot(target_point[0] - center[0], target_point[1] - center[1])
            score -= center_distance / box_diag
            if _point_in_bbox(target_point, bbox):
                score += 0.6

        if score > best_score:
            best_score = score
            best_track_id = int(track_id)

    return best_track_id


def _register_proxy_hand_to_mouth_event(
    config,
    person_track_id: int | None,
    event_active: bool,
    now_ts: float,
) -> tuple[bool, int]:
    """Track distinct per-person proxy gestures and trigger after settle delay."""
    state_by_person = getattr(config, "person_proxy_event_state", None)
    if not isinstance(state_by_person, dict):
        state_by_person = {}
        config.person_proxy_event_state = state_by_person

    def _prune_event_times(event_times: deque) -> None:
        while event_times and (now_ts - float(event_times[0])) > HAND_TO_MOUTH_WINDOW_SECONDS:
            event_times.popleft()

    def _evaluate_pending_trigger(state: dict) -> bool:
        pending_trigger_at = float(state.get("pending_trigger_at", 0.0))
        cooldown_until = float(state.get("cooldown_until", 0.0))
        if pending_trigger_at <= 0.0:
            return False
        if now_ts < pending_trigger_at or now_ts < cooldown_until:
            return False
        state["pending_trigger_at"] = 0.0
        state["cooldown_until"] = float(now_ts + HAND_TO_MOUTH_PERSON_COOLDOWN_SECONDS)
        state["last_trigger_ts"] = float(now_ts)
        return True

    if person_track_id is None:
        triggered = False
        max_count = 0
        for tracked_state in state_by_person.values():
            event_times = tracked_state.get("event_times")
            if not isinstance(event_times, deque):
                event_times = deque()
                tracked_state["event_times"] = event_times
            _prune_event_times(event_times)
            if bool(tracked_state.get("last_event_active", False)):
                tracked_state["last_event_active"] = False
                tracked_state["hand_away_since"] = float(now_ts)
            elif float(tracked_state.get("hand_away_since", 0.0)) <= 0.0:
                tracked_state["hand_away_since"] = float(now_ts)
            if _evaluate_pending_trigger(tracked_state):
                triggered = True
            max_count = max(max_count, len(event_times))

        stale_after = max(
            float(HAND_TO_MOUTH_PERSON_COOLDOWN_SECONDS) * 2.0,
            float(HAND_TO_MOUTH_PERSON_TRACK_MAX_GAP_SECONDS) * 8.0,
        )
        for tracked_person_id, tracked_state in list(state_by_person.items()):
            if (now_ts - float(tracked_state.get("last_seen_ts", 0.0))) > stale_after:
                state_by_person.pop(tracked_person_id, None)
        return triggered, max_count

    state = state_by_person.setdefault(
        int(person_track_id),
        {
            "event_times": deque(),
            "last_event_active": False,
            "cooldown_until": 0.0,
            "last_seen_ts": float(now_ts),
            "hand_away_since": float(now_ts - _PROXY_GESTURE_DEBOUNCE_SECONDS),
            "pending_trigger_at": 0.0,
            "last_logged_event_ts": float(now_ts - _PROXY_GESTURE_DEBOUNCE_SECONDS),
        },
    )
    event_times = state.get("event_times")
    if not isinstance(event_times, deque):
        event_times = deque()
        state["event_times"] = event_times

    _prune_event_times(event_times)

    last_event_active = bool(state.get("last_event_active", False))
    hand_away_since = float(state.get("hand_away_since", 0.0))
    cooldown_until = float(state.get("cooldown_until", 0.0))
    last_logged_event_ts = float(state.get("last_logged_event_ts", 0.0))
    if bool(event_active):
        if not last_event_active:
            away_duration = float("inf") if hand_away_since <= 0.0 else (now_ts - hand_away_since)
            log_debounce_elapsed = (now_ts - last_logged_event_ts) >= _PROXY_GESTURE_DEBOUNCE_SECONDS
            if (
                away_duration >= _PROXY_GESTURE_DEBOUNCE_SECONDS
                and log_debounce_elapsed
                and now_ts >= cooldown_until
            ):
                event_times.append(float(now_ts))
                state["last_logged_event_ts"] = float(now_ts)
        state["hand_away_since"] = 0.0
    else:
        if last_event_active or hand_away_since <= 0.0:
            state["hand_away_since"] = float(now_ts)

    _prune_event_times(event_times)

    event_count = len(event_times)
    pending_trigger_at = float(state.get("pending_trigger_at", 0.0))
    if (
        event_count >= int(HAND_TO_MOUTH_REQUIRED_EVENTS)
        and pending_trigger_at <= 0.0
        and now_ts >= cooldown_until
    ):
        state["pending_trigger_at"] = float(now_ts + _PROXY_ALERT_SETTLE_SECONDS)

    triggered = _evaluate_pending_trigger(state)

    state["last_event_active"] = bool(event_active)
    state["last_seen_ts"] = float(now_ts)

    stale_after = max(
        float(HAND_TO_MOUTH_PERSON_COOLDOWN_SECONDS) * 2.0,
        float(HAND_TO_MOUTH_PERSON_TRACK_MAX_GAP_SECONDS) * 8.0,
    )
    for tracked_person_id, tracked_state in list(state_by_person.items()):
        if (now_ts - float(tracked_state.get("last_seen_ts", 0.0))) > stale_after:
            state_by_person.pop(tracked_person_id, None)

    return triggered, event_count


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
    wrist_xy = getattr(config, "person_proxy_last_wrist_xy", None)
    if wrist_xy is None:
        wrist_xy = getattr(config, "person_proxy_last_finger_xy", None)
    if marker_bbox is None and mouth_xy is not None and wrist_xy is not None:
        min_x = min(float(mouth_xy[0]), float(wrist_xy[0]))
        min_y = min(float(mouth_xy[1]), float(wrist_xy[1]))
        max_x = max(float(mouth_xy[0]), float(wrist_xy[0]))
        max_y = max(float(mouth_xy[1]), float(wrist_xy[1]))
        span = max(max_x - min_x, max_y - min_y, 40.0)
        pad = max(26.0, span * 0.80)
        marker_bbox = (min_x - pad, min_y - pad, max_x + pad, max_y + pad)

    if marker_bbox is None and (mouth_xy is not None or wrist_xy is not None):
        anchor = mouth_xy if mouth_xy is not None else wrist_xy
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

    def _requires_strict_proxy_recording() -> bool:
        return bool(
            payload.get("video_required", False)
            and str(payload.get("alert_reason", "")).strip().lower() == "motion_burst"
            and str(payload.get("hand_to_mouth_source", "")).strip().lower() == "person_proxy"
        )

    def _cleanup_orphan_video_file(video_file_name: str) -> None:
        video_dir = str(payload.get("video_dir", "")).strip()
        if not video_dir:
            return
        safe_name = os.path.basename(str(video_file_name or "").strip())
        if not safe_name or safe_name in {".", ".."}:
            return
        candidate_path = os.path.abspath(os.path.join(video_dir, safe_name))
        try:
            if os.path.commonpath([os.path.abspath(video_dir), candidate_path]) != os.path.abspath(video_dir):
                return
        except ValueError:
            return
        try:
            if os.path.exists(candidate_path):
                os.remove(candidate_path)
        except OSError:
            pass

    def _resolve_safe_video_path(video_file_name: str) -> str:
        video_dir = str(payload.get("video_dir", "")).strip()
        if not video_dir:
            return ""
        safe_name = os.path.basename(str(video_file_name or "").strip())
        if not safe_name or safe_name in {".", ".."}:
            return ""
        root_abs = os.path.abspath(video_dir)
        candidate_path = os.path.abspath(os.path.join(root_abs, safe_name))
        try:
            if os.path.commonpath([root_abs, candidate_path]) != root_abs:
                return ""
        except ValueError:
            return ""
        return candidate_path

    def _cleanup_orphan_snippet_files(alert_record: dict) -> None:
        snippet_dir = str(payload.get("snippet_dir", "")).strip()
        if not snippet_dir or not isinstance(alert_record, dict):
            return
        detections = alert_record.get("detections")
        if not isinstance(detections, list):
            return
        snippet_root = os.path.abspath(snippet_dir)
        for det in detections:
            if not isinstance(det, dict):
                continue
            safe_name = os.path.basename(str(det.get("snippet_file", "")).strip())
            if not safe_name or safe_name in {".", ".."}:
                continue
            candidate_path = os.path.abspath(os.path.join(snippet_root, safe_name))
            try:
                if os.path.commonpath([snippet_root, candidate_path]) != snippet_root:
                    continue
            except ValueError:
                continue
            try:
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
            except OSError:
                pass

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
        video_required=payload.get("video_required", False),
        prefer_mp4=payload.get("prefer_mp4", False),
        alert_reason=payload["alert_reason"],
    )
    if _requires_strict_proxy_recording():
        if alert is None:
            logger.error(
                "Dropping proxy hand-to-mouth alert: required buffered recording was not created"
            )
            return
        video_file = str(alert.get("video_file", "")).strip()
        if not video_file:
            logger.error(
                "Dropping proxy hand-to-mouth alert: recording filename is missing"
            )
            _cleanup_orphan_video_file(video_file)
            _cleanup_orphan_snippet_files(alert)
            return
        video_path = _resolve_safe_video_path(video_file)
        video_size = -1
        if video_path and os.path.exists(video_path):
            try:
                video_size = int(os.path.getsize(video_path))
            except OSError:
                video_size = -1
        if video_size <= 0:
            logger.error(
                "Dropping proxy hand-to-mouth alert: buffered recording is empty or missing (size=%s)",
                video_size,
            )
            _cleanup_orphan_video_file(video_file)
            _cleanup_orphan_snippet_files(alert)
            return
    if alert is None:
        return
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
    ignored_class_ids: set[int] | None = None
    active_model = None
    active_allowed_names: frozenset[str] = frozenset()
    last_open_failure_summary = ""
    last_open_failure_logged_at = 0.0
    last_open_success_source = ""
    last_read_failure_logged_at = 0.0
    initial_buffer_frames = max(
        30,
        int(max(3.0, float(getattr(config, "stream_fps", 10.0))) * HAND_TO_MOUTH_VIDEO_BUFFER_SECONDS),
    )
    recent_frames: deque = deque(maxlen=initial_buffer_frames)

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

            target_buffer_frames = max(
                30,
                int(max(3.0, stream_fps) * HAND_TO_MOUTH_VIDEO_BUFFER_SECONDS),
            )
            if recent_frames.maxlen != target_buffer_frames:
                recent_frames = deque(recent_frames, maxlen=target_buffer_frames)

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                    active_cam_index = camera_index
                config.camera_available = False
                config.camera_error = "Camera is OFF"
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
                config.camera_available = False
                config.camera_error = f"Camera {camera_index} switching"
                allowed_ids = None
                next_inference_at = 0.0

            if cap is None:
                cap, source_label, diagnostics = _open_camera_capture_with_fallback(
                    camera_index,
                    width=out_width,
                    height=out_height,
                    stream_fps=stream_fps,
                )
                if cap is None:
                    summary = diagnostics[-1] if diagnostics else f"Camera {camera_index} unavailable"
                    config.camera_available = False
                    config.camera_error = f"Camera {camera_index} unavailable"
                    now_ts = time.time()
                    if (
                        summary != last_open_failure_summary
                        or (now_ts - last_open_failure_logged_at) >= 8.0
                    ):
                        logger.warning(
                            "Camera %s failed to open. Last attempt: %s",
                            camera_index,
                            summary,
                        )
                        for detail in diagnostics:
                            logger.warning("Camera %s open detail: %s", camera_index, detail)
                        last_open_failure_summary = summary
                        last_open_failure_logged_at = now_ts
                    _publish_status_frame(
                        config,
                        out_width,
                        out_height,
                        f"Camera {camera_index} unavailable",
                        jpeg_quality,
                    )
                    time.sleep(1.0)
                    continue
                config.camera_available = True
                config.camera_error = ""
                if source_label != last_open_success_source:
                    logger.info("Camera %s opened via %s", camera_index, source_label)
                    last_open_success_source = source_label
                active_cam_index = camera_index
                allowed_ids = None
                next_inference_at = 0.0

            wall_now = time.time()

            ok, frame = cap.read()
            if not ok:
                config.camera_available = False
                config.camera_error = f"Camera {camera_index} read failed"
                if (wall_now - last_read_failure_logged_at) >= 5.0:
                    logger.warning("Camera %s read failed, reconnecting", camera_index)
                    last_read_failure_logged_at = wall_now
                cap.release()
                cap = None
                allowed_ids = None
                recent_frames.clear()
                time.sleep(0.1)
                continue

            config.camera_available = True
            config.camera_error = ""

            # Cameras are physically mounted upside down; rotate before detection and streaming.
            frame = cv2.flip(frame, -1)

            frame_diag = math.hypot(float(frame.shape[1]), float(frame.shape[0]))

            recent_frames.append(frame.copy())

            detections = last_detections
            motion_detected = last_motion_detected
            motion_score = last_motion_score
            motion_source = "none"
            inference_ran = False
            hand_to_mouth_event_active = False
            hand_to_mouth_event_count = 0
            motion_burst_trigger = False
            burst_trigger_frames: list | None = None
            person_proxy_details: dict | None = None
            proxy_person_track_id: int | None = None
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
                        if model is not active_model:
                            active_model = model
                            ignored_class_ids = None
                            allowed_ids = None
                        if allowed_names_key != active_allowed_names:
                            active_allowed_names = allowed_names_key
                            allowed_ids = None
                        if ignored_class_ids is None:
                            ignored_class_ids = _ignored_yolo_class_ids(model)
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
                        if results:
                            result = results[0]
                            if ignored_class_ids:
                                _strip_result_boxes_by_class_ids(result, ignored_class_ids)
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
                    _update_proxy_person_tracks(config, person_detections, wall_now, frame_diag)
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
                        person_proxy_detected, person_proxy_score, person_proxy_details = detect_person_hand_to_mouth_proxy(
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
                        person_proxy_details = None

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

                    proxy_gesture_triggered = bool(
                        isinstance(person_proxy_details, dict)
                        and bool(person_proxy_details.get("gesture_triggered", False))
                    )
                    proxy_sequence_active = bool(
                        hand_to_mouth_event_active
                        and motion_source == "person_proxy"
                        and proxy_gesture_triggered
                    )
                    proxy_person_track_id = _resolve_proxy_person_track_id(
                        person_detections,
                        person_proxy_details,
                    )
                    motion_burst_trigger, hand_to_mouth_event_count = _register_proxy_hand_to_mouth_event(
                        config,
                        proxy_person_track_id,
                        proxy_sequence_active,
                        wall_now,
                    )
                    if motion_burst_trigger and motion_source != "person_proxy":
                        # Delayed burst triggers can fire after per-frame proxy score cools down.
                        # Keep the alert source consistent so downstream strict video checks apply.
                        motion_source = "person_proxy"
                        motion_detected = True
                        motion_score = max(float(motion_score), float(HAND_MOUTH_SCORE_FLOOR))
                    if motion_burst_trigger:
                        burst_trigger_frames = list(recent_frames)

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

            motion_burst_trigger = bool(inference_ran and motion_burst_trigger)
            hand_to_mouth_pending_sequence = (
                inference_ran
                and hand_to_mouth_event_active
                and motion_source == "person_proxy"
                and not motion_burst_trigger
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
                not hand_to_mouth_pending_sequence
                and not same_person_suppressed
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
                if motion_burst_trigger:
                    # Motion-burst alerts are hand-to-mouth events; keep payload class focused on that marker.
                    marker = _build_hand_to_mouth_alert_marker(
                        config,
                        frame,
                        person_detections,
                        motion_score,
                        motion_source,
                    )
                    if marker is not None:
                        payload_detections = [marker]
                    else:
                        payload_detections = [
                            det
                            for det in payload_detections
                            if isinstance(det, dict)
                            and normalize_class_label(str(det.get("class_name", ""))) == "hand_to_mouth"
                        ]

                is_hand_to_mouth_alert = any(
                    isinstance(det, dict)
                    and normalize_class_label(str(det.get("class_name", ""))) == "hand_to_mouth"
                    for det in payload_detections
                )

                # Only proxy hand-to-mouth burst alerts must carry video from the rolling buffer.
                # Static object alerts stay image-only.
                attach_video_for_alert = bool(motion_burst_trigger and is_hand_to_mouth_alert)
                video_required_for_alert = bool(motion_burst_trigger and is_hand_to_mouth_alert)
                should_persist_alert = True
                if motion_burst_trigger and not is_hand_to_mouth_alert:
                    should_persist_alert = False
                    logger.error(
                        "Dropping motion-burst alert: no hand_to_mouth detection in payload"
                    )
                if attach_video_for_alert:
                    recent_frames_snapshot = (
                        list(burst_trigger_frames)
                        if motion_burst_trigger and burst_trigger_frames
                        else list(recent_frames)
                    )
                    min_buffer_frames = max(
                        6,
                        int(round(max(3.0, float(stream_fps)) * _PROXY_MIN_BUFFER_SECONDS)),
                    )
                    if len(recent_frames_snapshot) < min_buffer_frames:
                        logger.error(
                            "Dropping proxy hand-to-mouth alert: buffered clip too short (%s < %s frames)",
                            len(recent_frames_snapshot),
                            min_buffer_frames,
                        )
                        should_persist_alert = False
                else:
                    recent_frames_snapshot = []

                if should_persist_alert:
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
                        "hand_to_mouth_event_count": int(hand_to_mouth_event_count),
                        "attach_video": attach_video_for_alert,
                        "video_required": video_required_for_alert,
                        "prefer_mp4": bool(motion_burst_trigger),
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
                        # Per-person cooldown throttles proxy burst alerts.
                        pass
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

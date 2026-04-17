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
    FOOD_CLASS_NAMES,
    FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE,
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
from dfx.motion import detect_consumption_motion, detect_person_hand_to_mouth_proxy


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
    config.person_proxy_prev_gray = None
    config.person_proxy_active_until = 0.0
    config.person_proxy_trigger_streak = 0
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

            recent_frames.append(frame.copy())

            detections = last_detections
            motion_detected = last_motion_detected
            motion_score = last_motion_score
            motion_source = "none"
            inference_ran = False
            all_detections = detections
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
                        if allowed_ids is None:
                            allowed_ids = get_allowed_class_ids(model, INFERENCE_CLASS_NAMES)
                        predict_kwargs = {
                            "verbose": False,
                            "conf": conf,
                            "iou": iou,
                            "imgsz": inference_imgsz,
                            "classes": allowed_ids if allowed_ids else None,
                            "device": getattr(config, "inference_device", "cpu"),
                        }
                        results = predict_with_fallback(model, frame, **predict_kwargs)
                if inference_ran:
                    result = results[0]
                    all_detections = detections_from_result(result, allowed_names=INFERENCE_CLASS_NAMES)
                    detections = [
                        det
                        for det in all_detections
                        if (
                            str(det.get("class_name", "")).strip().lower() in FOOD_CLASS_NAMES
                            and float(det.get("confidence", 0.0)) >= ALERT_DETECTION_CONFIDENCE_FLOOR
                        )
                    ]
                    person_detections = [
                        det
                        for det in all_detections
                        if str(det.get("class_name", "")).strip().lower() == "person"
                    ]
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
                        if not motion_detected and not detections and person_proxy_detected:
                            # Allow alerting on pure hand-to-mouth gesture only when no food object is visible.
                            motion_detected = True
                            motion_score = max(float(motion_score), float(person_proxy_score))
                            motion_source = "person_proxy"
                        if detections:
                            config.last_food_seen_ts = wall_now
                        if not motion_detected:
                            # Keep motion active briefly when food is momentarily occluded by a hand.
                            recently_saw_food = (
                                (wall_now - float(config.last_food_seen_ts)) <= FOOD_OCCLUSION_LOOKBACK_SECONDS
                            )
                            if (
                                not detections
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
                                and bool(detections)
                                and float(motion_score) >= FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE
                            )
                            or (
                                motion_source in {"person_proxy", "food_occluded"}
                                and not bool(detections)
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
            if detection_enabled and inference_ran and detections:
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
                and motion_detected
                and (bool(detections) or bool(person_detections))
                and len(config.motion_event_times) >= HAND_TO_MOUTH_REQUIRED_EVENTS
            )
            stationary_followup_trigger = (
                inference_ran
                and bool(detections)
                and not motion_detected
                and config.stationary_first_alert_ts > 0.0
                and not config.stationary_followup_sent
                and (wall_now - config.stationary_first_alert_ts) >= STATIONARY_FOLLOWUP_SECONDS
            )
            initial_trigger = (
                inference_ran
                and detections
                and config.consecutive >= max(1, persist_frames)
                and config.armed
                and (wall_now - config.last_alert_ts) >= max(0.0, cooldown)
            )
            frame_diag = math.hypot(float(frame.shape[1]), float(frame.shape[0]))
            new_object_trigger = (
                inference_ran
                and bool(detections)
                and (wall_now - config.last_alert_ts) >= NEW_OBJECT_MIN_ALERT_GAP_SECONDS
                and has_novel_alert_object(
                    config,
                    detections,
                    frame_diag=frame_diag,
                    now_ts=wall_now,
                )
            )

            same_person_suppressed = False
            alert_person_center = select_alert_person_center(person_detections, detections)
            same_person_suppressed = _is_same_person_suppressed(
                config,
                alert_person_center,
                wall_now,
                frame_diag,
            )

            should_alert = (
                not same_person_suppressed
                and (motion_burst_trigger or stationary_followup_trigger or initial_trigger or new_object_trigger)
            )
            if should_alert:
                reason = "initial"
                if motion_burst_trigger:
                    reason = "motion_burst"
                elif new_object_trigger:
                    reason = "new_object"
                elif stationary_followup_trigger:
                    reason = "stationary_followup"
                alert_payload = {
                    "frame": frame.copy(),
                    "detections": deepcopy(detections),
                    "snippet_dir": config.snippet_dir,
                    "video_dir": config.video_dir,
                    "recent_frames": list(recent_frames),
                    "video_fps": stream_fps,
                    "camera_zone": camera_zone,
                    "context_detections": deepcopy(all_detections),
                    "motion_detected": motion_detected,
                    "motion_score": motion_score,
                    "hand_to_mouth_source": motion_source,
                    "hand_to_mouth_event_count": len(config.motion_event_times),
                    "attach_video": (
                        motion_burst_trigger
                        and (bool(detections) or bool(person_detections))
                    ),
                    "alert_reason": reason,
                }
                if not _enqueue_alert_job(config, alert_payload):
                    _persist_alert_synchronously(config, alert_payload)
                config.last_alert_ts = wall_now
                remember_alert_objects(config, detections, wall_now)
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
            elif same_person_suppressed and motion_burst_trigger:
                # Prevent burst-alert loops for one person while still allowing future events.
                config.motion_event_times.clear()

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

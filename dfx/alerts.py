"""Alert persistence, snippet generation, video recording, and suppression logic."""

import csv
import json
import math
import os
import platform
from datetime import datetime
from uuid import uuid4

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

from dfx.constants import (
    ALERT_DETECTION_CONFIDENCE_FLOOR,
    ALERT_SNIPPET_CONFIDENCE_FLOOR,
    CONSUMPTION_CLASS_NAMES,
    DETECTION_SUMMARY_HEADERS,
    NEW_OBJECT_LOOKBACK_SECONDS,
    NEW_OBJECT_MATCH_DISTANCE_RATIO,
    NEW_OBJECT_MIN_CONFIDENCE,
)
from dfx.detection import safe_token
from dfx.motion import _extract_detection_geometry
from dfx.settings import normalize_camera_zone


def read_alerts(log_path: str | None) -> list[dict]:
    """Read the persisted alert list; invalid or missing files degrade to an empty list."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        return []
    return []


def write_alerts(log_path: str | None, alerts: list[dict]) -> None:
    """Persist the full alert list back to disk."""
    if not log_path:
        return
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(alerts, handle, indent=2)


def _split_alert_timestamp(value: str) -> tuple[str, str, str, str]:
    """Normalize one alert timestamp into CSV-friendly date and time columns."""
    raw = str(value or "").strip()
    if not raw:
        return "", "", "", ""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw, "", "", ""
    return (
        parsed.isoformat(timespec="seconds"),
        parsed.date().isoformat(),
        parsed.strftime("%A"),
        parsed.time().isoformat(timespec="seconds"),
    )


def append_detection_summary_csv(summary_path: str | None, alert: dict) -> None:
    """Append one CSV row per detection when a new alert is created."""
    if not summary_path or not isinstance(alert, dict):
        return
    detections = alert.get("detections")
    if not isinstance(detections, list) or not detections:
        return
    summary_path = os.path.abspath(summary_path)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    timestamp, date_value, weekday, time_value = _split_alert_timestamp(alert.get("timestamp", ""))
    rows: list[dict[str, str | float | bool]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        rows.append(
            {
                "alert_id": str(alert.get("id", "")).strip(),
                "timestamp": timestamp,
                "date": date_value,
                "weekday": weekday,
                "time": time_value,
                "zone": str(alert.get("zone", "")).strip(),
                "category": str(det.get("class_name", "")).strip().lower(),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "status": str(alert.get("status", "")).strip().lower() or "new",
                "consumption_motion_detected": bool(alert.get("consumption_motion_detected", False)),
                "consumption_motion_score": round(float(alert.get("consumption_motion_score", 0.0)), 3),
                "snippet_file": str(det.get("snippet_file", "")).strip(),
            }
        )
    if not rows:
        return
    with open(summary_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_SUMMARY_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def ensure_alert_metadata(alerts: list[dict]) -> bool:
    """Backfill IDs/status fields so old alert files still work with the current UI."""
    changed = False
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if not alert.get("id"):
            alert["id"] = uuid4().hex[:12]
            changed = True
        status = str(alert.get("status", "")).strip().lower()
        if status == "acknowledged":
            alert["status"] = "accepted"
            changed = True
        elif status not in {"new", "accepted"}:
            alert["status"] = "new"
            changed = True
    return changed


def alert_has_consumption_event(alert: dict) -> bool:
    """Treat one alert as one eating/drinking person event for backend counts."""
    if not isinstance(alert, dict):
        return False
    if bool(alert.get("consumption_motion_detected", False)):
        return True
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return False
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name in CONSUMPTION_CLASS_NAMES:
            return True
    return False


def _primary_consumption_category(alert: dict) -> str:
    """Pick one category per alert so table totals stay person-event based."""
    detections = alert.get("detections") if isinstance(alert, dict) else None
    if not isinstance(detections, list):
        return "unknown"
    best_name = "unknown"
    best_conf = -1.0
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name not in CONSUMPTION_CLASS_NAMES:
            continue
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= best_conf:
            best_conf = confidence
            best_name = class_name
    return best_name


def build_consumption_stats(alerts: list[dict]) -> dict:
    """Aggregate eating/drinking counts for the dashboard stats table."""
    total_people_detected = 0
    active_alerts = 0
    accepted_alerts = 0
    breakdown_counts: dict[tuple[str, str], int] = {}
    for alert in alerts:
        if not alert_has_consumption_event(alert):
            continue
        total_people_detected += 1
        status = str(alert.get("status", "")).strip().lower()
        if status == "accepted":
            accepted_alerts += 1
        else:
            active_alerts += 1
        zone = str(alert.get("zone", "")).strip() or "Unassigned"
        category = _primary_consumption_category(alert)
        key = (zone, category)
        breakdown_counts[key] = breakdown_counts.get(key, 0) + 1
    breakdown = [
        {"zone": zone, "category": category, "count": count}
        for (zone, category), count in sorted(
            breakdown_counts.items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    return {
        "total_people_detected": total_people_detected,
        "active_alerts": active_alerts,
        "accepted_alerts": accepted_alerts,
        "breakdown": breakdown,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def select_alert_person_center(person_detections: list[dict], detections: list[dict]) -> tuple[float, float] | None:
    """Choose one representative person center for duplicate-alert suppression."""
    if not person_detections:
        return None
    target_center = None
    if detections:
        sum_x = 0.0
        sum_y = 0.0
        count = 0
        for det in detections:
            center = det.get("center_xy") if isinstance(det, dict) else None
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                continue
            try:
                cx = float(center[0])
                cy = float(center[1])
            except (TypeError, ValueError):
                continue
            sum_x += cx
            sum_y += cy
            count += 1
        if count > 0:
            target_center = (sum_x / count, sum_y / count)

    best = None
    best_score = float("-inf")
    for det in person_detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        if target_center is None:
            score = confidence
        else:
            score = (confidence * 2.0) - math.hypot(center_x - target_center[0], center_y - target_center[1])
        if score > best_score:
            best_score = score
            best = (center_x, center_y)
    return best


def _iter_alert_object_candidates(detections: list[dict]):
    """Yield class/geometry tuples from detections for novelty checks."""
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if not class_name:
            continue
        geometry = _extract_detection_geometry(det)
        if geometry is None:
            continue
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        x, y, box_diag = geometry
        yield class_name, confidence, x, y, box_diag


def has_novel_alert_object(config, detections: list[dict], frame_diag: float, now_ts: float) -> bool:
    """Return True when at least one detection is a new object versus recent same-class alerts."""
    while (
        config.alert_object_history
        and (now_ts - float(config.alert_object_history[0][4])) > NEW_OBJECT_LOOKBACK_SECONDS
    ):
        config.alert_object_history.popleft()

    for class_name, confidence, x, y, box_diag in _iter_alert_object_candidates(detections):
        if confidence < NEW_OBJECT_MIN_CONFIDENCE:
            continue
        matched = False
        for prev_class, prev_x, prev_y, prev_diag, _prev_ts in config.alert_object_history:
            if prev_class != class_name:
                continue
            max_dist = max(
                float(frame_diag) * NEW_OBJECT_MATCH_DISTANCE_RATIO,
                float(box_diag) * 0.6,
                float(prev_diag) * 0.6,
            )
            if math.hypot(x - float(prev_x), y - float(prev_y)) <= max_dist:
                matched = True
                break
        if not matched:
            return True
    return False


def remember_alert_objects(config, detections: list[dict], now_ts: float) -> None:
    """Store recently alerted objects so we can distinguish repeats from genuinely new objects."""
    for class_name, confidence, x, y, box_diag in _iter_alert_object_candidates(detections):
        if confidence < ALERT_DETECTION_CONFIDENCE_FLOOR:
            continue
        config.alert_object_history.append((class_name, x, y, box_diag, float(now_ts)))


def clamp_box(bounds: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Clamp a float bbox into valid image coordinates."""
    x1, y1, x2, y2 = bounds
    left = max(0, min(frame_width - 1, int(round(float(x1)))))
    top = max(0, min(frame_height - 1, int(round(float(y1)))))
    right = max(left + 1, min(frame_width, int(round(float(x2)))))
    bottom = max(top + 1, min(frame_height, int(round(float(y2)))))
    return left, top, right, bottom


def _nearest_person_box(target_box: tuple[int, int, int, int], context_detections: list[dict]) -> tuple[int, int, int, int] | None:
    """Pick the person bbox nearest to the target item center."""
    tx1, ty1, tx2, ty2 = target_box
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    best_distance = float("inf")
    best_box = None
    for det in context_detections:
        if not isinstance(det, dict):
            continue
        if str(det.get("class_name", "")).strip().lower() != "person":
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        px1, py1, px2, py2 = (int(float(v)) for v in bbox)
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        distance = math.hypot(target_cx - pcx, target_cy - pcy)
        if distance < best_distance:
            best_distance = distance
            best_box = (px1, py1, px2, py2)
    return best_box


def add_detection_snippets(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    alert_id: str,
    context_detections: list[dict] | None = None,
):
    """Save contextual crops with item/person framing and attach filenames to detections."""
    if not snippet_dir:
        return detections
    os.makedirs(snippet_dir, exist_ok=True)
    height, width = frame.shape[:2]
    context_detections = context_detections or detections
    for idx, det in enumerate(detections):
        try:
            det_confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            det_confidence = 0.0
        if det_confidence < ALERT_SNIPPET_CONFIDENCE_FLOOR:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        item_left, item_top, item_right, item_bottom = clamp_box(list(bbox), width, height)
        person_box = _nearest_person_box(
            (item_left, item_top, item_right, item_bottom),
            context_detections,
        )

        crop_left, crop_top, crop_right, crop_bottom = item_left, item_top, item_right, item_bottom
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            crop_left = min(crop_left, max(0, px1))
            crop_top = min(crop_top, max(0, py1))
            crop_right = max(crop_right, min(width, px2))
            crop_bottom = max(crop_bottom, min(height, py2))

        box_w = max(1, crop_right - crop_left)
        box_h = max(1, crop_bottom - crop_top)
        margin_x = max(16, int(box_w * 0.2))
        margin_y = max(16, int(box_h * 0.2))
        crop_left = max(0, crop_left - margin_x)
        crop_top = max(0, crop_top - margin_y)
        crop_right = min(width, crop_right + margin_x)
        crop_bottom = min(height, crop_bottom + margin_y)

        crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()
        if crop.size == 0:
            continue

        local_item_left = max(0, item_left - crop_left)
        local_item_top = max(0, item_top - crop_top)
        local_item_right = max(local_item_left + 1, item_right - crop_left)
        local_item_bottom = max(local_item_top + 1, item_bottom - crop_top)
        cv2.rectangle(crop, (local_item_left, local_item_top), (local_item_right, local_item_bottom), (0, 180, 255), 2)
        class_name = str(det.get("class_name", "item"))
        cv2.putText(
            crop,
            class_name,
            (local_item_left, max(16, local_item_top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            local_px1 = max(0, px1 - crop_left)
            local_py1 = max(0, py1 - crop_top)
            local_px2 = max(local_px1 + 1, px2 - crop_left)
            local_py2 = max(local_py1 + 1, py2 - crop_top)
            cv2.rectangle(crop, (local_px1, local_py1), (local_px2, local_py2), (48, 195, 110), 2)

        class_token = safe_token(det.get("class_name", "item"))
        snippet_file = f"{alert_id}_{idx}_{class_token}.jpg"
        snippet_path = os.path.join(snippet_dir, snippet_file)
        if cv2.imwrite(snippet_path, crop):
            det["snippet_file"] = snippet_file
            crop_h, crop_w = crop.shape[:2]
            cx = ((local_item_left + local_item_right) / 2.0) / max(1.0, float(crop_w))
            cy = ((local_item_top + local_item_bottom) / 2.0) / max(1.0, float(crop_h))
            bw = (local_item_right - local_item_left) / max(1.0, float(crop_w))
            bh = (local_item_bottom - local_item_top) / max(1.0, float(crop_h))
            det["snippet_bbox_xywhn"] = [
                round(max(0.0, min(1.0, cx)), 6),
                round(max(0.0, min(1.0, cy)), 6),
                round(max(0.0, min(1.0, bw)), 6),
                round(max(0.0, min(1.0, bh)), 6),
            ]
    return detections


def add_alert_video(
    recent_frames: list,
    video_dir: str | None,
    alert_id: str,
    fps: float,
) -> tuple[str, str] | None:
    """Persist a short alert clip captured around the trigger moment."""
    if cv2 is None:
        return None
    if not video_dir or not recent_frames:
        return None
    os.makedirs(video_dir, exist_ok=True)

    first = recent_frames[0]
    if first is None or not hasattr(first, "shape") or len(first.shape) < 2:
        return None
    height, width = int(first.shape[0]), int(first.shape[1])
    if width <= 0 or height <= 0:
        return None
    safe_fps = max(3.0, float(fps or 8.0))

    def _is_usable_alert_video(path: str, min_written_frames: int) -> bool:
        """Accept only files that are non-trivial and decodable for browser playback."""
        if not os.path.exists(path):
            return False
        # Tiny files are often invalid headers with no real media payload.
        if os.path.getsize(path) < 4096:
            return False
        capture = cv2.VideoCapture(path)
        if not capture or not capture.isOpened():
            return False
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok, first_frame = capture.read()
        finally:
            capture.release()
        return bool(ok and first_frame is not None and frame_count >= max(2, min_written_frames // 2))

    def _write_alert_gif() -> tuple[str, str] | None:
        """Fallback for environments where browser-friendly video codecs are unavailable."""
        if Image is None:
            return None
        valid_frames = [
            frame
            for frame in recent_frames
            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2
        ]
        if len(valid_frames) < 2:
            return None
        max_frames = 28
        sample_step = max(1, len(valid_frames) // max_frames)
        sampled_frames = valid_frames[::sample_step]
        if len(sampled_frames) < 2:
            sampled_frames = [valid_frames[0], valid_frames[-1]]

        pil_frames = []
        for frame in sampled_frames:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
            if frame_h <= 0 or frame_w <= 0:
                continue
            target_width = min(480, frame_w)
            if frame_w != target_width:
                target_height = max(1, int(round(frame_h * (target_width / float(frame_w)))))
                frame = cv2.resize(frame, (target_width, target_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
        if len(pil_frames) < 2:
            return None

        output_name = f"{alert_id}.gif"
        output_path = os.path.join(video_dir, output_name)
        frame_interval_ms = int(round(1000.0 / max(3.0, min(15.0, safe_fps))))
        duration_ms = max(55, min(500, frame_interval_ms * sample_step))
        try:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
        except Exception:
            return None
        if os.path.exists(output_path) and os.path.getsize(output_path) >= 256:
            return output_name, "image/gif"
        try:
            os.remove(output_path)
        except OSError:
            pass
        return None

    if platform.system() == "Linux":
        # Jetson/OpenCV builds frequently emit MP4 files that decode in OpenCV but fail in browsers.
        # Prefer an animated GIF attachment so the dashboard always renders the recording inline.
        gif_result = _write_alert_gif()
        if gif_result is not None:
            return gif_result

    # Prefer the most reliably available codec first on Linux/Jetson to reduce encoder-probe noise.
    if platform.system() == "Linux":
        codec_candidates = [
            ("mp4v", "mp4", "video/mp4"),
            ("avc1", "mp4", "video/mp4"),
            ("H264", "mp4", "video/mp4"),
            ("X264", "mp4", "video/mp4"),
            ("VP80", "webm", "video/webm"),
            ("VP90", "webm", "video/webm"),
        ]
    else:
        codec_candidates = [
            ("avc1", "mp4", "video/mp4"),
            ("H264", "mp4", "video/mp4"),
            ("X264", "mp4", "video/mp4"),
            ("VP90", "webm", "video/webm"),
            ("VP80", "webm", "video/webm"),
            ("mp4v", "mp4", "video/mp4"),
        ]
    for codec, extension, mime in codec_candidates:
        output_name = f"{alert_id}.{extension}"
        output_path = os.path.join(video_dir, output_name)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            (width, height),
        )
        if not writer or not writer.isOpened():
            continue
        written_frames = 0
        try:
            for frame in recent_frames:
                if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
                    continue
                frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
                if frame_h != height or frame_w != width:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                written_frames += 1
        finally:
            writer.release()
        if written_frames <= 0:
            try:
                os.remove(output_path)
            except OSError:
                pass
            continue
        if _is_usable_alert_video(output_path, min_written_frames=written_frames):
            if platform.system() == "Linux" and codec == "mp4v":
                # MPEG-4 Part 2 often decodes in OpenCV but not in browser <video>.
                gif_result = _write_alert_gif()
                if gif_result is not None:
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    return gif_result
            return output_name, mime
        try:
            os.remove(output_path)
        except OSError:
            pass
    gif_result = _write_alert_gif()
    if gif_result is not None:
        return gif_result
    return None


def create_alert(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    video_dir: str | None,
    recent_frames: list | None,
    video_fps: float,
    camera_zone: str,
    context_detections: list[dict] | None = None,
    motion_detected: bool = False,
    motion_score: float = 0.0,
    hand_to_mouth_source: str = "none",
    hand_to_mouth_event_count: int = 0,
    attach_video: bool = False,
    alert_reason: str = "standard",
) -> dict:
    """Build the alert record stored in JSON and rendered by the dashboard."""
    alert_id = uuid4().hex[:12]
    zone = normalize_camera_zone(camera_zone)
    for det in detections:
        det["zone"] = zone
    video_file = None
    video_mime = ""
    if motion_detected and attach_video:
        video_result = add_alert_video(
            recent_frames=recent_frames or [],
            video_dir=video_dir,
            alert_id=alert_id,
            fps=video_fps,
        )
        if video_result is not None:
            video_file, video_mime = video_result
    return {
        "id": alert_id,
        "status": "new",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "alert_reason": str(alert_reason or "standard"),
        "zone": zone,
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "consumption_motion_detected": bool(motion_detected),
        "consumption_motion_score": round(float(motion_score), 3),
        "hand_to_mouth_source": str(hand_to_mouth_source or "none"),
        "hand_to_mouth_event_count": int(max(0, hand_to_mouth_event_count)),
        "video_file": video_file,
        "video_mime": video_mime,
        "detections": add_detection_snippets(
            frame,
            detections,
            snippet_dir,
            alert_id,
            context_detections=context_detections,
        ),
    }


def append_alert(log_path: str | None, alert: dict, summary_csv_path: str | None = None) -> None:
    """Append one alert while preserving compatibility metadata."""
    alerts = read_alerts(log_path)
    ensure_alert_metadata(alerts)
    alerts.append(alert)
    write_alerts(log_path, alerts)
    append_detection_summary_csv(summary_csv_path, alert)

"""YOLO result parsing, drawing helpers, and shared detection utilities."""

import random
from datetime import datetime, timedelta
from uuid import uuid4

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from dfx.constants import CAMERA_ZONES, FOOD_CLASS_NAMES


def get_allowed_class_ids(model, allowed_names: set[str]) -> list[int]:
    """Map human-readable class names to the integer class IDs exposed by YOLO."""
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, list):
        items = enumerate(names)
    else:
        return []
    allowed: list[int] = []
    for cls_id, name in items:
        if name and name.strip().lower() in allowed_names:
            allowed.append(int(cls_id))
    return sorted(allowed)


def detections_from_result(result, allowed_names: set[str] | None = None) -> list[dict]:
    """Normalize one YOLO prediction result into plain JSON-serializable dicts."""
    detections: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    names = result.names
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[idx])
        conf = float(result.boxes.conf[idx])
        cls_id = int(result.boxes.cls[idx])
        class_name = names.get(cls_id, str(cls_id))
        normalized = class_name.strip().lower()
        if allowed_names and normalized not in allowed_names:
            continue
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            }
        )
    return detections


def draw_detections(frame, detections):
    """Overlay bounding boxes and labels onto a frame for streaming to the browser."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        if det.get("consumption_motion"):
            label = f"{label} motion"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )
    return annotated


def safe_token(value: str) -> str:
    """Convert a free-form label into a filesystem-safe token."""
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    token = token.strip("_")
    return token or "item"


def make_placeholder_svg(width: int, height: int, label: str) -> str:
    """Build a lightweight placeholder image for test mode or missing camera states."""
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '  <rect width="100%" height="100%" fill="#121824" />\n'
        '  <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" '
        'font-family="Arial, sans-serif" font-size="32" fill="#f3f4f6">'
        f"{safe_label}</text>\n"
        "</svg>\n"
    )


def make_status_frame(width: int, height: int, label: str):
    """Create a simple text frame shown when the real camera feed is unavailable/off."""
    if cv2 is None or np is None:
        return None
    safe_width = max(320, int(width or 640))
    safe_height = max(180, int(height or 360))
    frame = np.zeros((safe_height, safe_width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        label,
        (24, safe_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
    )
    return frame


def make_random_alerts(limit: int, frame_width: int, frame_height: int) -> list[dict]:
    """Generate synthetic alerts so the UI can be exercised without a live camera."""
    alerts: list[dict] = []
    class_names = sorted(FOOD_CLASS_NAMES)
    for idx in range(max(1, limit)):
        det_count = random.randint(1, 3)
        detections = []
        zone = random.choice(CAMERA_ZONES)
        for _ in range(det_count):
            class_name = random.choice(class_names)
            x1 = random.randint(0, max(0, frame_width - 60))
            y1 = random.randint(0, max(0, frame_height - 60))
            x2 = random.randint(x1 + 20, min(frame_width, x1 + 200))
            y2 = random.randint(y1 + 20, min(frame_height, y1 + 200))
            detections.append(
                {
                    "class_id": -1,
                    "class_name": class_name,
                    "confidence": round(random.uniform(0.5, 0.99), 2),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                    "zone": zone,
                }
            )
        alert_time = datetime.now() - timedelta(seconds=idx * 3)
        alerts.append(
            {
                "id": uuid4().hex[:12],
                "status": random.choice(["new", "accepted"]),
                "timestamp": alert_time.isoformat(timespec="seconds"),
                "zone": zone,
                "frame_size": {"width": frame_width, "height": frame_height},
                "detections": detections,
            }
        )
    return alerts

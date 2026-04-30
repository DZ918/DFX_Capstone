"""Background OpenAI vision checks, persistence helpers, and dashboard payloads."""

from __future__ import annotations

import base64
import glob
import json
import logging
import os
import time
from datetime import datetime
from urllib.parse import quote

try:
    import cv2
except Exception:
    cv2 = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from dfx.constants import FOOD_CLASS_NAMES

logger = logging.getLogger(__name__)

DEFAULT_ADVANCED_DETECTION_MODEL = "gpt-4.1-mini"
DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS = 300
DEFAULT_ADVANCED_DETECTION_OUTPUT_DIR = "advanced_detections"
ALLOWED_ITEM_TYPES = {
    "packaged_food",
    "drink",
    "container",
    "wrapper",
    "eating_behavior",
    "drinking_behavior",
    "unknown",
}
FALLBACK_FOOD_TYPES = {
    "unknown_food",
    "unknown_drink",
    "unknown_packaged_food",
    "ambiguous",
}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def camera_id_for_index(camera_index: int) -> str:
    return f"camera_{int(camera_index)}"


def snapshot_from_frame(camera_index: int, frame, *, timestamp: str | None = None) -> dict:
    """Build one OpenAI-vision snapshot payload from a raw frame."""
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    return {
        "camera_index": int(camera_index),
        "camera_id": camera_id_for_index(camera_index),
        "timestamp": timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
        "width": int(width),
        "height": int(height),
        "frame": frame_copy,
    }


def configure_advanced_detection(config, *, test_mode: bool = False) -> None:
    output_dir = os.path.abspath(
        os.environ.get(
            "ADVANCED_DETECTION_OUTPUT_DIR",
            DEFAULT_ADVANCED_DETECTION_OUTPUT_DIR,
        ).strip()
        or DEFAULT_ADVANCED_DETECTION_OUTPUT_DIR
    )
    interval_raw = os.environ.get(
        "ADVANCED_DETECTION_INTERVAL_SECONDS",
        str(DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS),
    ).strip()
    try:
        interval_seconds = max(30, int(float(interval_raw)))
    except ValueError:
        interval_seconds = DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS

    config.advanced_detection_enabled = bool(
        env_bool("ADVANCED_DETECTION_ENABLED", default=False) and not bool(test_mode)
    )
    config.advanced_detection_interval_seconds = int(interval_seconds)
    config.advanced_detection_model = (
        os.environ.get("ADVANCED_DETECTION_MODEL", "").strip()
        or DEFAULT_ADVANCED_DETECTION_MODEL
    )
    config.advanced_detection_output_dir = output_dir
    config.advanced_detection_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    config.advanced_detection_running = False
    config.advanced_detection_last_started_at = ""
    config.advanced_detection_last_completed_at = ""
    config.advanced_detection_last_error = ""
    config.advanced_detection_last_message = (
        "Disabled"
        if not config.advanced_detection_enabled
        else "Waiting for the first interval"
    )
    config.advanced_detection_next_run_at = 0.0
    if hasattr(config, "default_settings") and isinstance(config.default_settings, dict):
        config.default_settings.update(
            {
                "advanced_detection_enabled": bool(config.advanced_detection_enabled),
                "advanced_detection_interval_seconds": int(config.advanced_detection_interval_seconds),
                "advanced_detection_model": str(config.advanced_detection_model),
                "advanced_detection_output_dir": str(config.advanced_detection_output_dir),
            }
        )


def advanced_detection_status_snapshot(config) -> dict:
    status_lock = getattr(config, "advanced_detection_lock", None)
    if status_lock is None:
        return {}
    with status_lock:
        return {
            "enabled": bool(getattr(config, "advanced_detection_enabled", False)),
            "running": bool(getattr(config, "advanced_detection_running", False)),
            "interval_seconds": int(
                getattr(
                    config,
                    "advanced_detection_interval_seconds",
                    DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS,
                )
            ),
            "model": str(getattr(config, "advanced_detection_model", "")),
            "output_dir": str(getattr(config, "advanced_detection_output_dir", "")),
            "last_started_at": str(getattr(config, "advanced_detection_last_started_at", "")),
            "last_completed_at": str(getattr(config, "advanced_detection_last_completed_at", "")),
            "last_error": str(getattr(config, "advanced_detection_last_error", "")),
            "message": str(getattr(config, "advanced_detection_last_message", "")),
            "next_run_at": float(getattr(config, "advanced_detection_next_run_at", 0.0)),
        }


def _update_status(config, **changes) -> None:
    status_lock = getattr(config, "advanced_detection_lock", None)
    if status_lock is None:
        return
    with status_lock:
        for key, value in changes.items():
            setattr(config, key, value)


def _advanced_detection_schema() -> dict:
    bbox_schema = {
        "type": "object",
        "properties": {
            "x1": {"type": "integer"},
            "y1": {"type": "integer"},
            "x2": {"type": "integer"},
            "y2": {"type": "integer"},
        },
        "required": ["x1", "y1", "x2", "y2"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "advanced_food_detection",
        "schema": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "advanced_detection_present": {"type": "boolean"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "food_type": {"type": "string"},
                            "description": {"type": "string"},
                            "bbox": bbox_schema,
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "type",
                            "food_type",
                            "description",
                            "bbox",
                            "confidence",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
                "subjects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "bbox": bbox_schema,
                            "confidence": {"type": "number"},
                            "related_items": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["description", "bbox", "confidence", "related_items"],
                        "additionalProperties": False,
                    },
                },
                "overall_summary": {"type": "string"},
            },
            "required": [
                "camera_id",
                "timestamp",
                "advanced_detection_present",
                "items",
                "subjects",
                "overall_summary",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _build_prompt(snapshot: dict, food_classes: list[str]) -> str:
    class_list = ", ".join(food_classes) if food_classes else ", ".join(sorted(FOOD_CLASS_NAMES))
    return (
        "Analyze this camera frame for food or drinks.\n\n"
        "Look for:\n"
        "- packaged food\n"
        "- snack bags\n"
        "- wrappers\n"
        "- cups, bottles, cans\n"
        "- eating or drinking behavior\n"
        "- ambiguous objects\n\n"
        f"Camera ID: {snapshot['camera_id']}\n"
        f"Timestamp: {snapshot['timestamp']}\n"
        f"Image size: {snapshot['width']}x{snapshot['height']}\n\n"
        "Use this food class list exactly when an item clearly matches:\n"
        f"{class_list}\n\n"
        "If no class matches, use one of:\n"
        "- unknown_food\n"
        "- unknown_drink\n"
        "- unknown_packaged_food\n"
        "- ambiguous\n\n"
        "Return JSON only.\n"
        "Use integer pixel bounding boxes (x1, y1, x2, y2) relative to the image.\n"
        "If unsure, lower confidence rather than invent precision.\n"
    )


def _encode_frame_data_url(frame) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not ok:
        raise RuntimeError("Could not encode frame for OpenAI Vision")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _clamp_bbox(bbox, width: int, height: int) -> dict:
    if not isinstance(bbox, dict):
        bbox = {}
    try:
        x1 = int(round(float(bbox.get("x1", 0))))
        y1 = int(round(float(bbox.get("y1", 0))))
        x2 = int(round(float(bbox.get("x2", width))))
        y2 = int(round(float(bbox.get("y2", height))))
    except (TypeError, ValueError):
        x1, y1, x2, y2 = 0, 0, max(1, width), max(1, height)
    left = max(0, min(max(0, width - 1), x1))
    top = max(0, min(max(0, height - 1), y1))
    right = max(left + 1, min(max(1, width), x2))
    bottom = max(top + 1, min(max(1, height), y2))
    return {"x1": left, "y1": top, "x2": right, "y2": bottom}


def _normalize_item_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALLOWED_ITEM_TYPES else "unknown"


def _normalize_food_type(value: str, item_type: str, allowed_food_types: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in allowed_food_types:
        return normalized
    if normalized in FALLBACK_FOOD_TYPES:
        return normalized
    if item_type in {"drink", "container", "drinking_behavior"}:
        return "unknown_drink"
    if item_type in {"packaged_food", "wrapper"}:
        return "unknown_packaged_food"
    if item_type == "eating_behavior":
        return "unknown_food"
    return "ambiguous"


def _normalize_items(items, width: int, height: int, allowed_food_types: set[str]) -> list[dict]:
    normalized_items: list[dict] = []
    if not isinstance(items, list):
        return normalized_items
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = _normalize_item_type(item.get("type", "unknown"))
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized_items.append(
            {
                "type": item_type,
                "food_type": _normalize_food_type(
                    item.get("food_type", ""),
                    item_type,
                    allowed_food_types,
                ),
                "description": str(item.get("description", "")).strip(),
                "bbox": _clamp_bbox(item.get("bbox"), width, height),
                "confidence": round(confidence, 4),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return normalized_items


def _normalize_subjects(subjects, width: int, height: int, item_count: int) -> list[dict]:
    normalized_subjects: list[dict] = []
    if not isinstance(subjects, list):
        return normalized_subjects
    for subject in subjects:
        if not isinstance(subject, dict):
            continue
        try:
            confidence = max(0.0, min(1.0, float(subject.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        related_items = []
        for item_index in subject.get("related_items", []):
            try:
                normalized_index = int(item_index)
            except (TypeError, ValueError):
                continue
            if 0 <= normalized_index < item_count:
                related_items.append(normalized_index)
        normalized_subjects.append(
            {
                "description": str(subject.get("description", "")).strip(),
                "bbox": _clamp_bbox(subject.get("bbox"), width, height),
                "confidence": round(confidence, 4),
                "related_items": related_items,
            }
        )
    return normalized_subjects


def _annotated_frame(frame, record: dict):
    if cv2 is None:
        return frame
    annotated = frame.copy()
    item_colors = {
        "packaged_food": (35, 190, 255),
        "drink": (255, 180, 60),
        "container": (255, 135, 20),
        "wrapper": (255, 80, 120),
        "eating_behavior": (60, 220, 120),
        "drinking_behavior": (180, 120, 255),
        "unknown": (180, 180, 180),
    }
    for item in record.get("items", []):
        bbox = item.get("bbox", {})
        left = int(bbox.get("x1", 0))
        top = int(bbox.get("y1", 0))
        right = int(bbox.get("x2", 0))
        bottom = int(bbox.get("y2", 0))
        color = item_colors.get(item.get("type", "unknown"), (180, 180, 180))
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{item.get('food_type', item.get('type', 'item'))} {float(item.get('confidence', 0.0)):.2f}"
        cv2.putText(
            annotated,
            label[:72],
            (left, max(18, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
    for subject in record.get("subjects", []):
        bbox = subject.get("bbox", {})
        left = int(bbox.get("x1", 0))
        top = int(bbox.get("y1", 0))
        right = int(bbox.get("x2", 0))
        bottom = int(bbox.get("y2", 0))
        cv2.rectangle(annotated, (left, top), (right, bottom), (82, 255, 168), 1)
    return annotated


def _relative_asset_urls(record: dict) -> dict:
    payload = dict(record)
    image_path = str(record.get("image_path", "")).strip().replace(os.sep, "/")
    annotated_path = str(record.get("annotated_image_path", "")).strip().replace(os.sep, "/")
    json_path = str(record.get("json_path", "")).strip().replace(os.sep, "/")
    if image_path:
        payload["image_url"] = f"/advanced-detections/assets/{quote(image_path)}"
    if annotated_path:
        payload["annotated_image_url"] = f"/advanced-detections/assets/{quote(annotated_path)}"
    if json_path:
        payload["json_url"] = f"/advanced-detections/assets/{quote(json_path)}"
    return payload


def _persist_record(output_dir: str, snapshot: dict, record: dict) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
    date_dir = snapshot["timestamp"][:10]
    timestamp_token = snapshot["timestamp"].replace(":", "-")
    rel_dir = os.path.join(date_dir, snapshot["camera_id"])
    abs_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    rel_image_path = os.path.join(rel_dir, f"{timestamp_token}.jpg")
    rel_annotated_path = os.path.join(rel_dir, f"{timestamp_token}_annotated.jpg")
    rel_json_path = os.path.join(rel_dir, f"{timestamp_token}.json")

    abs_image_path = os.path.join(output_dir, rel_image_path)
    abs_annotated_path = os.path.join(output_dir, rel_annotated_path)
    abs_json_path = os.path.join(output_dir, rel_json_path)

    cv2.imwrite(abs_image_path, snapshot["frame"])
    cv2.imwrite(abs_annotated_path, _annotated_frame(snapshot["frame"], record))

    record_to_save = dict(record)
    record_to_save["image_path"] = rel_image_path
    record_to_save["annotated_image_path"] = rel_annotated_path
    record_to_save["json_path"] = rel_json_path
    with open(abs_json_path, "w", encoding="utf-8") as handle:
        json.dump(record_to_save, handle, indent=2)
    return _relative_asset_urls(record_to_save)


def _collect_snapshots(config) -> list[dict]:
    snapshots: list[dict] = []
    primary_frame = getattr(config, "latest_frame", None)
    with config.settings_lock:
        primary_index = int(config.camera_index)
    if primary_frame is not None:
        snapshots.append(snapshot_from_frame(primary_index, primary_frame))

    camera_manager = getattr(config, "camera_manager", None)
    if camera_manager is None:
        return snapshots
    for preview_snapshot in camera_manager.snapshot_frames():
        frame = preview_snapshot.get("frame")
        if frame is None:
            continue
        camera_index = int(preview_snapshot["camera_index"])
        snapshots.append(snapshot_from_frame(camera_index, frame))
    return snapshots


def _build_openai_client(config):
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed")
    api_key = str(getattr(config, "advanced_detection_api_key", "")).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def run_advanced_detection_once(
    config,
    snapshots: list[dict],
    *,
    trigger_label: str = "manual",
    update_next_run: bool = False,
    block_if_busy: bool = False,
) -> list[dict]:
    """Run OpenAI advanced detection immediately for one or more prepared snapshots."""
    if not isinstance(snapshots, list) or not snapshots:
        raise RuntimeError("No camera frames available yet")

    run_lock = getattr(config, "advanced_detection_run_lock", None)
    lock_acquired = False
    if run_lock is not None:
        lock_acquired = run_lock.acquire(blocking=bool(block_if_busy))
        if not lock_acquired:
            raise RuntimeError("Advanced detection is already running")
    try:
        client = _build_openai_client(config)
        output_dir = str(
            getattr(config, "advanced_detection_output_dir", DEFAULT_ADVANCED_DETECTION_OUTPUT_DIR)
        )
        model = str(getattr(config, "advanced_detection_model", DEFAULT_ADVANCED_DETECTION_MODEL))
        interval_seconds = int(
            getattr(
                config,
                "advanced_detection_interval_seconds",
                DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS,
            )
        )
        allowed_food_types = {
            str(name).strip().lower()
            for name in getattr(config, "runtime_food_class_names", FOOD_CLASS_NAMES)
            if str(name).strip()
        }
        run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        _update_status(
            config,
            advanced_detection_running=True,
            advanced_detection_last_started_at=run_started_at,
            advanced_detection_last_error="",
            advanced_detection_last_message=f"{trigger_label}: processing {len(snapshots)} frame(s)",
        )

        persisted_records: list[dict] = []
        for snapshot in snapshots:
            try:
                record = _call_openai_detection(client, model, snapshot, allowed_food_types)
            except Exception as exc:
                logger.warning(
                    "Advanced detection failed for %s: %s",
                    snapshot.get("camera_id", "camera"),
                    exc,
                )
                record = _error_record(snapshot, str(exc))
                _update_status(
                    config,
                    advanced_detection_last_error=str(exc),
                    advanced_detection_last_message=f"{trigger_label}: API error on {snapshot['camera_id']}",
                )
            try:
                persisted_records.append(_persist_record(output_dir, snapshot, record))
            except Exception as exc:
                logger.warning(
                    "Could not persist advanced detection for %s: %s",
                    snapshot.get("camera_id", "camera"),
                    exc,
                )
                _update_status(
                    config,
                    advanced_detection_last_error=str(exc),
                    advanced_detection_last_message=f"{trigger_label}: persistence error on {snapshot['camera_id']}",
                )

        status_updates = {
            "advanced_detection_running": False,
            "advanced_detection_last_completed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "advanced_detection_last_message": f"{trigger_label}: complete",
        }
        if update_next_run:
            status_updates["advanced_detection_next_run_at"] = time.time() + interval_seconds
        _update_status(config, **status_updates)
        return persisted_records
    finally:
        if lock_acquired:
            run_lock.release()


def _call_openai_detection(client, model: str, snapshot: dict, allowed_food_types: set[str]) -> dict:
    data_url = _encode_frame_data_url(snapshot["frame"])
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _build_prompt(snapshot, sorted(allowed_food_types))},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
        text={"format": _advanced_detection_schema()},
    )
    if not getattr(response, "output_text", ""):
        raise RuntimeError("OpenAI Vision returned no structured output")
    payload = json.loads(response.output_text)
    items = _normalize_items(
        payload.get("items", []),
        snapshot["width"],
        snapshot["height"],
        allowed_food_types,
    )
    subjects = _normalize_subjects(
        payload.get("subjects", []),
        snapshot["width"],
        snapshot["height"],
        len(items),
    )
    summary = str(payload.get("overall_summary", "")).strip()
    return {
        "camera_id": snapshot["camera_id"],
        "timestamp": snapshot["timestamp"],
        "image_width": int(snapshot["width"]),
        "image_height": int(snapshot["height"]),
        "advanced_detection_present": bool(items),
        "items": items,
        "subjects": subjects,
        "overall_summary": summary or ("Items detected" if items else "No advanced detections"),
        "api_error": "",
    }


def _error_record(snapshot: dict, error_message: str) -> dict:
    return {
        "camera_id": snapshot["camera_id"],
        "timestamp": snapshot["timestamp"],
        "image_width": int(snapshot["width"]),
        "image_height": int(snapshot["height"]),
        "advanced_detection_present": False,
        "items": [],
        "subjects": [],
        "overall_summary": "API error",
        "api_error": str(error_message).strip() or "API error",
    }


def advanced_detection_worker(config) -> None:
    while not getattr(config, "stop", False):
        enabled = bool(getattr(config, "advanced_detection_enabled", False))
        if not enabled:
            _update_status(
                config,
                advanced_detection_running=False,
                advanced_detection_last_message="Advanced detection disabled",
            )
            time.sleep(1.0)
            continue

        interval_seconds = int(
            getattr(
                config,
                "advanced_detection_interval_seconds",
                DEFAULT_ADVANCED_DETECTION_INTERVAL_SECONDS,
            )
        )
        next_run_at = float(getattr(config, "advanced_detection_next_run_at", 0.0))
        now = time.time()
        if next_run_at > now:
            _update_status(
                config,
                advanced_detection_running=False,
                advanced_detection_last_message="Waiting for next interval",
            )
            time.sleep(min(1.0, max(0.1, next_run_at - now)))
            continue

        snapshots = _collect_snapshots(config)
        if not snapshots:
            _update_status(
                config,
                advanced_detection_running=False,
                advanced_detection_last_message="No camera frames available yet",
                advanced_detection_next_run_at=now + interval_seconds,
            )
            time.sleep(1.0)
            continue

        try:
            run_advanced_detection_once(
                config,
                snapshots,
                trigger_label="scheduled",
                update_next_run=True,
                block_if_busy=True,
            )
        except Exception as exc:
            _update_status(
                config,
                advanced_detection_running=False,
                advanced_detection_last_error=str(exc),
                advanced_detection_last_message="Advanced detection unavailable",
                advanced_detection_next_run_at=time.time() + interval_seconds,
            )
            time.sleep(1.0)


def read_recent_advanced_detections(output_dir: str, limit: int = 20) -> list[dict]:
    if not output_dir:
        return []
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        return []
    pattern = os.path.join(output_dir, "*", "*", "*.json")
    paths = glob.glob(pattern)
    paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    detections: list[dict] = []
    for path in paths[: max(0, int(limit))]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                detections.append(_relative_asset_urls(payload))
        except (OSError, json.JSONDecodeError):
            continue
    detections.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return detections[: max(0, int(limit))]


def _safe_output_path(output_dir: str, relative_path: str) -> str:
    root = os.path.abspath(output_dir)
    normalized = str(relative_path or "").strip().lstrip("/").replace("/", os.sep)
    candidate = os.path.abspath(os.path.join(root, normalized))
    if not candidate.startswith(f"{root}{os.sep}"):
        raise ValueError("Invalid advanced detection path")
    return candidate


def acknowledge_advanced_detection(output_dir: str, json_path: str) -> dict:
    """Delete one persisted advanced-detection JSON record and its image assets."""
    if not output_dir:
        raise ValueError("Advanced detection storage is unavailable")
    record_path = _safe_output_path(output_dir, json_path)
    if not os.path.exists(record_path):
        raise FileNotFoundError("Advanced detection record not found")

    rel_paths: set[str] = set()
    try:
        with open(record_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("image_path", "annotated_image_path", "json_path"):
                value = str(payload.get(key, "")).strip()
                if value:
                    rel_paths.add(value)
    except (OSError, json.JSONDecodeError):
        rel_paths.add(json_path)

    deleted = []
    for rel_path in rel_paths:
        abs_path = _safe_output_path(output_dir, rel_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
                deleted.append(rel_path)
            except OSError as exc:
                raise RuntimeError(f"Could not remove '{rel_path}': {exc}") from exc
    return {"deleted": deleted}

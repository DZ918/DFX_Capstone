"""Runtime configuration, settings validation, and the shared DashboardConfig state object."""

import os
import threading
from collections import deque
from datetime import datetime


def parse_bool(value, field_name: str) -> bool:
    """Accept a few JSON-friendly boolean representations from the settings API."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean for '{field_name}'")


def normalize_camera_zone(value, field_name: str = "camera_zone") -> str:
    """Validate and normalize the configured camera zone label."""
    zone = str(value).strip().upper()
    if zone.startswith("ZONE "):
        zone = zone[5:].strip()
    if len(zone) == 1 and zone in "ABCDEFGHI":
        return f"Zone {zone}"
    raise ValueError(f"Invalid value for '{field_name}'. Expected Zone A through Zone I.")


def clamp_float(value, field_name: str, minimum: float, maximum: float) -> float:
    """Parse and bound a float setting so runtime updates stay within safe limits."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid number for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def clamp_int(value, field_name: str, minimum: int, maximum: int) -> int:
    """Parse and bound an integer setting so runtime updates stay within safe limits."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def normalize_text(value, field_name: str, *, allow_empty: bool = False, maximum_length: int = 256) -> str:
    """Validate a generic short text setting."""
    text = str(value).strip()
    if not text and not allow_empty:
        raise ValueError(f"Invalid value for '{field_name}'")
    if len(text) > maximum_length:
        raise ValueError(f"Value for '{field_name}' is too long")
    return text


def settings_snapshot(config) -> dict:
    """Return the live runtime settings exposed to the dashboard."""
    with config.settings_lock:
        return {
            "camera_enabled": bool(config.camera_enabled),
            "detection_enabled": bool(config.detection_enabled),
            "motion_enabled": bool(config.motion_enabled),
            "conf": float(config.conf),
            "iou": float(config.iou),
            "persist_frames": int(config.persist_frames),
            "cooldown": float(config.cooldown),
            "clear_frames": int(config.clear_frames),
            "stream_fps": float(config.stream_fps),
            "width": int(config.width),
            "height": int(config.height),
            "inference_imgsz": int(config.inference_imgsz),
            "max_inference_fps": float(config.max_inference_fps),
            "jpeg_quality": int(config.jpeg_quality),
            "motion_hold_seconds": float(config.motion_hold_seconds),
            "camera_index": int(config.camera_index),
            "camera_zone": str(config.camera_zone),
            "advanced_detection_enabled": bool(getattr(config, "advanced_detection_enabled", False)),
            "advanced_detection_interval_seconds": int(
                getattr(config, "advanced_detection_interval_seconds", 300)
            ),
            "advanced_detection_model": str(getattr(config, "advanced_detection_model", "")),
            "advanced_detection_output_dir": str(
                getattr(config, "advanced_detection_output_dir", "advanced_detections")
            ),
            "updated_at": config.settings_updated_at,
            "test_mode": bool(config.test_mode),
        }


def default_settings_snapshot(config) -> dict:
    """Return the original startup settings so the UI can restore defaults."""
    with config.settings_lock:
        defaults = dict(config.default_settings)
    defaults["test_mode"] = bool(config.test_mode)
    return defaults


def update_runtime_settings(config, payload: dict) -> dict:
    """Apply validated settings updates atomically while the camera thread is running."""
    if not isinstance(payload, dict):
        raise ValueError("Settings payload must be a JSON object")
    with config.settings_lock:
        if "camera_enabled" in payload:
            config.camera_enabled = parse_bool(payload["camera_enabled"], "camera_enabled")
        if "detection_enabled" in payload:
            config.detection_enabled = parse_bool(payload["detection_enabled"], "detection_enabled")
        if "motion_enabled" in payload:
            config.motion_enabled = parse_bool(payload["motion_enabled"], "motion_enabled")
        if "conf" in payload:
            config.conf = clamp_float(payload["conf"], "conf", 0.01, 1.0)
        if "iou" in payload:
            config.iou = clamp_float(payload["iou"], "iou", 0.01, 1.0)
        if "persist_frames" in payload:
            config.persist_frames = clamp_int(payload["persist_frames"], "persist_frames", 1, 120)
        if "cooldown" in payload:
            config.cooldown = clamp_float(payload["cooldown"], "cooldown", 0.0, 3600.0)
        if "clear_frames" in payload:
            config.clear_frames = clamp_int(payload["clear_frames"], "clear_frames", 1, 600)
        if "stream_fps" in payload:
            config.stream_fps = clamp_float(payload["stream_fps"], "stream_fps", 1.0, 60.0)
        if "width" in payload:
            config.width = clamp_int(payload["width"], "width", 0, 3840)
        if "height" in payload:
            config.height = clamp_int(payload["height"], "height", 0, 2160)
        if "inference_imgsz" in payload:
            # Keep runtime updates in a small-object-safe range for wide-FOV cameras.
            config.inference_imgsz = clamp_int(payload["inference_imgsz"], "inference_imgsz", 960, 1280)
        if "max_inference_fps" in payload:
            config.max_inference_fps = clamp_float(
                payload["max_inference_fps"], "max_inference_fps", 0.0, 60.0
            )
        if "jpeg_quality" in payload:
            config.jpeg_quality = clamp_int(payload["jpeg_quality"], "jpeg_quality", 40, 95)
        if "motion_hold_seconds" in payload:
            config.motion_hold_seconds = clamp_float(
                payload["motion_hold_seconds"], "motion_hold_seconds", 0.0, 5.0
            )
        if "camera_index" in payload:
            config.camera_index = clamp_int(payload["camera_index"], "camera_index", 0, 32)
        if "camera_zone" in payload:
            config.camera_zone = normalize_camera_zone(payload["camera_zone"])
        if "advanced_detection_enabled" in payload:
            config.advanced_detection_enabled = parse_bool(
                payload["advanced_detection_enabled"], "advanced_detection_enabled"
            )
            config.advanced_detection_next_run_at = 0.0
        if "advanced_detection_interval_seconds" in payload:
            config.advanced_detection_interval_seconds = clamp_int(
                payload["advanced_detection_interval_seconds"],
                "advanced_detection_interval_seconds",
                30,
                86400,
            )
            config.advanced_detection_next_run_at = 0.0
        if "advanced_detection_model" in payload:
            config.advanced_detection_model = normalize_text(
                payload["advanced_detection_model"],
                "advanced_detection_model",
                maximum_length=128,
            )
            config.advanced_detection_next_run_at = 0.0
        if "advanced_detection_output_dir" in payload:
            config.advanced_detection_output_dir = os.path.abspath(
                normalize_text(
                    payload["advanced_detection_output_dir"],
                    "advanced_detection_output_dir",
                    maximum_length=512,
                )
            )
        config.settings_updated_at = datetime.now().isoformat(timespec="seconds")
    return settings_snapshot(config)


def reset_runtime_settings(config) -> dict:
    """Reset mutable runtime settings back to their startup defaults."""
    defaults = default_settings_snapshot(config)
    defaults.pop("test_mode", None)
    return update_runtime_settings(config, defaults)


class DashboardConfig:
    """Shared mutable state for the HTTP handlers, camera loop, and training worker."""
    def __init__(
        self,
        model,
        model_path,
        alert_log,
        camera_index,
        width,
        height,
        stream_fps,
        conf,
        iou,
        persist_frames,
        cooldown,
        clear_frames,
        camera_zone,
        map_image_path,
        snippet_dir,
        detection_summary_csv,
        inference_imgsz,
        max_inference_fps,
        jpeg_quality,
        motion_hold_seconds,
        training_dir,
        train_epochs,
        train_imgsz,
        motion_enabled,
        motion_window,
        motion_displacement_threshold,
        motion_upward_threshold,
        test_mode,
    ):
        self.model = model
        self.model_path = model_path
        self.alert_log = alert_log
        self.camera_index = int(camera_index)
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.conf = conf
        self.iou = iou
        self.persist_frames = persist_frames
        self.cooldown = cooldown
        self.clear_frames = clear_frames
        self.camera_zone = normalize_camera_zone(camera_zone)
        if os.path.isabs(map_image_path):
            self.map_image_path = os.path.abspath(map_image_path)
        else:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.abspath(os.path.join(package_dir, ".."))
            candidate_paths = [
                os.path.abspath(os.path.join(workspace_root, map_image_path)),
                os.path.abspath(os.path.join(package_dir, map_image_path)),
                os.path.abspath(map_image_path),
            ]
            self.map_image_path = next(
                (candidate for candidate in candidate_paths if os.path.exists(candidate)),
                candidate_paths[0],
            )
        self.camera_enabled = True
        self.detection_enabled = True
        self.settings_updated_at = datetime.now().isoformat(timespec="seconds")
        self.snippet_dir = snippet_dir
        self.video_dir = (
            os.path.join(os.path.abspath(snippet_dir), "videos") if snippet_dir else None
        )
        self.detection_summary_csv = (
            os.path.abspath(detection_summary_csv) if detection_summary_csv else None
        )
        self.latest_frame = None
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.training_lock = threading.Lock()
        self.default_settings = {
            "camera_enabled": True,
            "detection_enabled": True,
            "conf": float(conf),
            "iou": float(iou),
            "persist_frames": int(persist_frames),
            "cooldown": float(cooldown),
            "clear_frames": int(clear_frames),
            "stream_fps": float(stream_fps),
            "width": int(width),
            "height": int(height),
            "inference_imgsz": int(inference_imgsz),
            "max_inference_fps": float(max_inference_fps),
            "jpeg_quality": int(jpeg_quality),
            "motion_hold_seconds": float(motion_hold_seconds),
            "camera_index": int(camera_index),
            "camera_zone": self.camera_zone,
            "motion_enabled": bool(motion_enabled),
            "advanced_detection_enabled": bool(getattr(self, "advanced_detection_enabled", False)),
            "advanced_detection_interval_seconds": int(
                getattr(self, "advanced_detection_interval_seconds", 300)
            ),
            "advanced_detection_model": str(getattr(self, "advanced_detection_model", "")),
            "advanced_detection_output_dir": str(
                getattr(self, "advanced_detection_output_dir", "advanced_detections")
            ),
        }
        self.stop = False
        self.consecutive = 0
        self.clear_count = 0
        self.armed = True
        self.last_alert_ts = 0.0
        self.stationary_first_alert_ts = 0.0
        self.stationary_followup_sent = False
        self.motion_event_times: deque[float] = deque()
        self.last_motion_active = False
        self.last_food_seen_ts = 0.0
        self.occlusion_motion_until = 0.0
        self.person_proxy_prev_gray = None
        self.person_proxy_landmark_detector = None
        self.person_proxy_landmark_detector_unavailable = False
        self.person_proxy_active_until = 0.0
        self.person_proxy_score_history: deque[float] = deque(maxlen=6)
        self.person_proxy_trigger_streak = 0
        self.person_proxy_dwell_started_at = 0.0
        self.person_proxy_last_seen_ts = 0.0
        self.person_proxy_last_approach_ts = 0.0
        self.person_proxy_last_distance_ratio = float("inf")
        self.person_proxy_last_finger_xy = None
        self.person_proxy_last_mouth_xy = None
        self.food_motion_confirm_streak = 0
        self.person_alert_history: deque[tuple[float, float, float]] = deque(maxlen=300)
        self.alert_object_history: deque[tuple[str, float, float, float, float]] = deque(maxlen=500)
        self.test_mode = test_mode
        self.training_dir = os.path.abspath(training_dir)
        self.training_data_dir = os.path.join(self.training_dir, "dataset")
        self.training_images_dir = os.path.join(self.training_data_dir, "images")
        self.training_labels_dir = os.path.join(self.training_data_dir, "labels")
        self.training_yaml_path = os.path.join(self.training_data_dir, "data.yaml")
        self.class_map_path = os.path.join(self.training_dir, "class_map.json")
        self.training_runs_dir = os.path.join(self.training_dir, "runs")
        self.train_epochs = int(train_epochs)
        self.train_imgsz = int(train_imgsz)
        self.inference_imgsz = int(inference_imgsz)
        self.max_inference_fps = float(max_inference_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.motion_hold_seconds = float(motion_hold_seconds)
        self.motion_enabled = bool(motion_enabled)
        self.motion_window = max(4, int(motion_window))
        self.motion_displacement_threshold = float(motion_displacement_threshold)
        self.motion_upward_threshold = float(motion_upward_threshold)
        self.motion_tracks: dict[int, dict] = {}
        self.next_motion_track_id = 1
        self.training_thread = None
        self.training_running = False
        self.training_last_started_at = ""
        self.training_last_completed_at = ""
        self.training_last_error = ""
        self.training_last_message = ""
        self.training_last_weights = ""
        os.makedirs(self.training_dir, exist_ok=True)

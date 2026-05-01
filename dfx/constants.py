"""Shared constants for class names, thresholds, and scoring parameters."""

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINED_DASHBOARD_MODEL_PATH = os.path.join(
    _PROJECT_ROOT,
    "training_data",
    "runs",
    "accepted",
    "weights",
    "best.pt",
)
_BASE_DASHBOARD_MODEL_PATH = os.path.join(_PROJECT_ROOT, "yolov8n.pt")


def _default_dashboard_model_path() -> str:
    configured = os.environ.get("DFX_DASHBOARD_MODEL_PATH", "").strip()
    if configured:
        return configured
    if os.path.exists(_TRAINED_DASHBOARD_MODEL_PATH):
        return _TRAINED_DASHBOARD_MODEL_PATH
    return _BASE_DASHBOARD_MODEL_PATH if os.path.exists(_BASE_DASHBOARD_MODEL_PATH) else "yolov8n.pt"


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return int(default)
    try:
        return int(raw_value)
    except ValueError:
        return int(default)


def _read_text_env(name: str, default: str) -> str:
    raw_value = os.environ.get(name, "").strip()
    return raw_value or str(default)


def normalize_class_label(value: str) -> str:
    """Normalize class labels for consistent matching across models and config files."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


_IGNORED_YOLO_CLASS_NAMES = {
    "hand_to_mouth",
    "handtomouth",
}


def is_ignored_yolo_class_name(value: str) -> bool:
    """Return whether a class name should be ignored during YOLO inference parsing."""
    return normalize_class_label(value) in _IGNORED_YOLO_CLASS_NAMES


DEFAULT_DASHBOARD_MODEL_PATH = _default_dashboard_model_path()
DEFAULT_DASHBOARD_CONFIDENCE = float(os.environ.get("DFX_DASHBOARD_CONFIDENCE", "0.35"))

DASHBOARD_PRIMARY_CAMERA_INDEX = _read_int_env("DFX_DASHBOARD_CAMERA_1_INDEX", 0)
DASHBOARD_SECONDARY_CAMERA_INDEX = _read_int_env("DFX_DASHBOARD_CAMERA_2_INDEX", 4)
if DASHBOARD_SECONDARY_CAMERA_INDEX == DASHBOARD_PRIMARY_CAMERA_INDEX:
    DASHBOARD_SECONDARY_CAMERA_INDEX = DASHBOARD_PRIMARY_CAMERA_INDEX + 1

DASHBOARD_PRIMARY_CAMERA_LABEL = _read_text_env("DFX_DASHBOARD_CAMERA_1_LABEL", "Camera 1")
DASHBOARD_SECONDARY_CAMERA_LABEL = _read_text_env("DFX_DASHBOARD_CAMERA_2_LABEL", "Camera 2")
DASHBOARD_PRIMARY_CAMERA_ZONE = _read_text_env("DFX_DASHBOARD_CAMERA_1_ZONE", "Zone G")
DASHBOARD_SECONDARY_CAMERA_ZONE = _read_text_env("DFX_DASHBOARD_CAMERA_2_ZONE", "Zone F")

SNACK_CLASS_NAMES = {
    "snack",
    "snacks",
    "lays",
    "doritos",
    "candy",
    "candies",
    "candy wrapper",
    "candy wrappers",
    "candy_wrapper",
    "candy_wrappers",
    "vending machine food",
    "vending_machine_food",
    "drink",
    "drinks",
}

FOOD_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "bottle",
    "cup",
    "bowl",
} | SNACK_CLASS_NAMES

INFERENCE_CLASS_NAMES = set(FOOD_CLASS_NAMES) | {"person"}

CONSUMPTION_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "snack",
    "snacks",
    "lays",
    "doritos",
    "candy",
    "candies",
    "vending machine food",
    "vending_machine_food",
    "drink",
    "drinks",
    "bottle",
    "cup",
}

DRINK_CONTAINER_CLASS_NAMES = {"bottle", "cup", "drink", "drinks"}
HANDHELD_FOOD_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "snack",
    "snacks",
    "lays",
    "doritos",
    "candy",
    "candies",
    "vending machine food",
    "vending_machine_food",
}

# Motion scoring thresholds.
MOTION_TRIGGER_SCORE = 0.85
FOOD_MOTION_MIN_SCORE = 1.0
FOOD_MOTION_CONFIRM_FRAMES = 3
FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE = 1.12
PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE = 1.2
STATIONARY_FOLLOWUP_SECONDS = 30 * 60
HAND_TO_MOUTH_WINDOW_SECONDS = 30.0
HAND_TO_MOUTH_REQUIRED_EVENTS = 3
HAND_TO_MOUTH_PERSON_COOLDOWN_SECONDS = 60.0
HAND_TO_MOUTH_VIDEO_BUFFER_SECONDS = 10.0
HAND_TO_MOUTH_PERSON_TRACK_MAX_GAP_SECONDS = 2.0
HAND_TO_MOUTH_PERSON_TRACK_MATCH_DISTANCE_RATIO = 0.16
FOOD_OCCLUSION_LOOKBACK_SECONDS = 2.0
OCCLUDED_MOTION_HOLD_SECONDS = 1.2
OCCLUDED_MOTION_PROXY_SCORE = 0.86
HAND_TO_MOUTH_FOOD_VISIBILITY_FLOOR = 0.45

# Landmark-based hand-to-mouth detection parameters.
HAND_MOUTH_MIN_PERSON_CONFIDENCE = 0.25
HAND_MOUTH_MIN_PERSON_AREA_RATIO = 0.005
HAND_MOUTH_PERSON_CROP_MARGIN_RATIO = 0.26
HAND_MOUTH_LANDMARK_MIN_DETECTION_CONFIDENCE = 0.45
HAND_MOUTH_LANDMARK_MIN_TRACKING_CONFIDENCE = 0.45
HAND_MOUTH_MIN_BBOX_SCALE_PX = 18.0
HAND_MOUTH_MAX_DISTANCE_RATIO = 0.92
HAND_MOUTH_MIN_DWELL_SECONDS = 0.26
HAND_MOUTH_MAX_TRACK_GAP_SECONDS = 0.45
HAND_MOUTH_APPROACH_WINDOW_SECONDS = 0.70
HAND_MOUTH_MIN_APPROACH_DELTA_RATIO = 0.0075
HAND_MOUTH_MIN_DIRECTION_COSINE = 0.05
HAND_MOUTH_LANDMARK_EMA_ALPHA = 0.33
HAND_MOUTH_HOLD_SECONDS = 0.60
HAND_MOUTH_SCORE_FLOOR = 1.24
HAND_MOUTH_WRIST_UPWARD_HISTORY_SECONDS = 0.80
HAND_MOUTH_MIN_WRIST_UPWARD_RATIO = 0.035
HAND_MOUTH_MIN_WRIST_UPWARD_STEPS = 2

# Alert confidence floors.
ALERT_DETECTION_CONFIDENCE_FLOOR = 0.30
ALERT_SNIPPET_CONFIDENCE_FLOOR = 0.25

# Paths and limits.
DEFAULT_MAP_IMAGE_PATH = os.path.join("assets", "dfx_lab_map.png")
TRAIN_VIDEO_SAMPLE_MAX_FRAMES = 12

# Same-person suppression.
SAME_PERSON_ALERT_WINDOW_SECONDS = 120.0
SAME_PERSON_MAX_ALERTS_IN_WINDOW = 3
SAME_PERSON_SUPPRESSION_DISTANCE_RATIO = 0.18

# New-object detection.
NEW_OBJECT_LOOKBACK_SECONDS = 45.0
NEW_OBJECT_MATCH_DISTANCE_RATIO = 0.1
NEW_OBJECT_MIN_ALERT_GAP_SECONDS = 1.0
NEW_OBJECT_MIN_CONFIDENCE = 0.68

CAMERA_ZONES = tuple(f"Zone {chr(ord('A') + idx)}" for idx in range(9))

DETECTION_SUMMARY_HEADERS = (
    "alert_id",
    "timestamp",
    "date",
    "weekday",
    "time",
    "zone",
    "category",
    "confidence",
    "status",
    "consumption_motion_detected",
    "consumption_motion_score",
    "snippet_file",
)

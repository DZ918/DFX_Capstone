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


DEFAULT_DASHBOARD_MODEL_PATH = _default_dashboard_model_path()
DEFAULT_DASHBOARD_CONFIDENCE = float(os.environ.get("DFX_DASHBOARD_CONFIDENCE", "0.35"))

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
PROXY_HAND_TO_MOUTH_REQUIRED_EVENTS = 3
FOOD_OCCLUSION_LOOKBACK_SECONDS = 2.0
OCCLUDED_MOTION_HOLD_SECONDS = 1.2
OCCLUDED_MOTION_PROXY_SCORE = 0.86
HAND_TO_MOUTH_FOOD_VISIBILITY_FLOOR = 0.45

# Landmark-based hand-to-mouth detection parameters.
HAND_MOUTH_MIN_PERSON_CONFIDENCE = 0.25
HAND_MOUTH_MIN_PERSON_AREA_RATIO = 0.02
HAND_MOUTH_PERSON_CROP_MARGIN_RATIO = 0.18
HAND_MOUTH_LANDMARK_MIN_DETECTION_CONFIDENCE = 0.55
HAND_MOUTH_LANDMARK_MIN_TRACKING_CONFIDENCE = 0.55
HAND_MOUTH_MIN_FACE_WIDTH_PX = 15.0
HAND_MOUTH_MAX_DISTANCE_RATIO = 0.30
HAND_MOUTH_MIN_DWELL_SECONDS = 0.20
HAND_MOUTH_MAX_TRACK_GAP_SECONDS = 0.28
HAND_MOUTH_APPROACH_WINDOW_SECONDS = 0.45
HAND_MOUTH_MIN_APPROACH_DELTA_RATIO = 0.006
HAND_MOUTH_MIN_DIRECTION_COSINE = 0.12
HAND_MOUTH_LANDMARK_EMA_ALPHA = 0.45
HAND_MOUTH_HOLD_SECONDS = 0.45
HAND_MOUTH_SCORE_FLOOR = 1.28

# Alert confidence floors.
ALERT_DETECTION_CONFIDENCE_FLOOR = 0.62
ALERT_SNIPPET_CONFIDENCE_FLOOR = 0.64

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

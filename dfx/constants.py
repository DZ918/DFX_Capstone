"""Shared constants for class names, thresholds, and scoring parameters."""

import os

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
}

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
    "bottle",
    "cup",
}

DRINK_CONTAINER_CLASS_NAMES = {"bottle", "cup"}
HANDHELD_FOOD_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
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
FOOD_OCCLUSION_LOOKBACK_SECONDS = 2.0
OCCLUDED_MOTION_HOLD_SECONDS = 1.2
OCCLUDED_MOTION_PROXY_SCORE = 0.86

# Person-proxy hand-to-mouth detection parameters.
PERSON_PROXY_MIN_AREA_RATIO = 0.02
PERSON_PROXY_DIFF_THRESHOLD = 24
PERSON_PROXY_MOUTH_MOTION_RATIO = 0.05
PERSON_PROXY_MIN_MOUTH_RATIO = 0.04
PERSON_PROXY_HOLD_SECONDS = 0.8
PERSON_PROXY_SCORE_FLOOR = 0.86
PERSON_PROXY_APPROACH_MOTION_RATIO = 0.035
PERSON_PROXY_MIN_APPROACH_RATIO = 0.028
PERSON_PROXY_TRIGGER_SCORE = 1.1
PERSON_PROXY_CONFIRM_FRAMES = 3
PERSON_PROXY_MIN_CONFIDENCE = 0.25

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

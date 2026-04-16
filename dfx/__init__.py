"""DFX Lab Food/Drink Detection and Monitoring System."""

from dfx.constants import FOOD_CLASS_NAMES, INFERENCE_CLASS_NAMES
from dfx.detection import get_allowed_class_ids, detections_from_result
from dfx.gpu import get_best_device

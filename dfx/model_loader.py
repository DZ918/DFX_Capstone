"""Inference model loading helpers, including optional base+trained model fusion."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dfx.constants import INFERENCE_CLASS_NAMES

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO, YOLOWorld
except Exception:
    YOLO = None
    YOLOWorld = None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_MODEL_PATH = _PROJECT_ROOT / "yolov8n.pt"
_TRAINED_MODEL_DIR_FRAGMENT = os.path.join("training_data", "runs", "accepted", "weights")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def _iter_model_names(model: Any) -> list[tuple[int, str]]:
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return [(int(class_id), str(name)) for class_id, name in names.items()]
    if isinstance(names, list):
        return [(int(class_id), str(name)) for class_id, name in enumerate(names)]
    return []


def _parse_bool_env(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def _resolve_base_merge_path(model_path: str, base_model_path: str | None = None) -> str | None:
    disable_merge = _parse_bool_env("DFX_DISABLE_MODEL_MERGE")
    if disable_merge is True:
        return None

    configured_base = str(base_model_path or os.environ.get("DFX_BASE_MODEL_PATH", "")).strip()
    candidate_base = Path(configured_base) if configured_base else _DEFAULT_BASE_MODEL_PATH
    if not candidate_base.exists():
        return None

    try:
        normalized_model = Path(model_path).resolve(strict=False)
        normalized_base = candidate_base.resolve(strict=False)
    except OSError:
        return None
    if normalized_model == normalized_base:
        return None

    force_merge = _parse_bool_env("DFX_MERGE_BASE_MODEL")
    if force_merge is False:
        return None

    model_basename = normalized_model.name.lower()
    model_text = str(normalized_model)
    auto_merge = (
        model_basename == "best.pt"
        or _TRAINED_MODEL_DIR_FRAGMENT in model_text
    )
    if force_merge is not True and not auto_merge:
        return None
    return str(normalized_base)


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = max(1e-6, area_a + area_b - inter_area)
    return inter_area / union


def _classwise_nms(
    detections: list[tuple[float, float, float, float, float, int]],
    iou_threshold: float,
) -> list[tuple[float, float, float, float, float, int]]:
    if len(detections) <= 1:
        return detections

    grouped: dict[int, list[tuple[float, float, float, float, float, int]]] = {}
    for detection in detections:
        grouped.setdefault(int(detection[5]), []).append(detection)

    kept: list[tuple[float, float, float, float, float, int]] = []
    for class_id, class_detections in grouped.items():
        ranked = sorted(class_detections, key=lambda item: float(item[4]), reverse=True)
        class_kept: list[tuple[float, float, float, float, float, int]] = []
        for detection in ranked:
            bbox = detection[:4]
            if all(_iou(bbox, existing[:4]) <= iou_threshold for existing in class_kept):
                class_kept.append(detection)
        kept.extend(class_kept)
    return sorted(kept, key=lambda item: float(item[4]), reverse=True)


class _CombinedBoxes:
    def __init__(self, detections: list[tuple[float, float, float, float, float, int]]):
        self.xyxy = [[det[0], det[1], det[2], det[3]] for det in detections]
        self.conf = [float(det[4]) for det in detections]
        self.cls = [int(det[5]) for det in detections]

    def __len__(self) -> int:
        return len(self.conf)


class _CombinedResult:
    def __init__(self, detections: list[tuple[float, float, float, float, float, int]], names: dict[int, str]):
        self.boxes = _CombinedBoxes(detections)
        self.names = names


class CombinedYOLOModel:
    """Run a base model and a trained model, then merge detections into one result."""

    def __init__(self, models: list[Any], labels: list[str] | None = None):
        self._models = list(models)
        self._labels = list(labels or [])
        self.names: dict[int, str] = {}
        self._per_model_class_map: list[dict[int, int]] = []
        self._dfx_inference_device_override = ""

        normalized_to_combined: dict[str, int] = {}
        for model in self._models:
            class_map: dict[int, int] = {}
            for class_id, class_name in _iter_model_names(model):
                normalized = _normalize_name(class_name)
                if not normalized:
                    continue
                combined_id = normalized_to_combined.get(normalized)
                if combined_id is None:
                    combined_id = len(self.names)
                    normalized_to_combined[normalized] = combined_id
                    self.names[combined_id] = str(class_name).strip()
                class_map[int(class_id)] = combined_id
            self._per_model_class_map.append(class_map)

    def to(self, device: str):
        for model in self._models:
            for candidate in (model, getattr(model, "model", None)):
                if candidate is None or not hasattr(candidate, "to"):
                    continue
                try:
                    candidate.to(device)
                    break
                except Exception:
                    continue
        self._dfx_inference_device_override = str(device)
        return self

    def predict(self, source: Any, **kwargs):
        allowed_combined_ids = kwargs.get("classes")
        allowed_set = None
        if allowed_combined_ids is not None:
            allowed_set = {int(class_id) for class_id in allowed_combined_ids}

        merged_detections: list[tuple[float, float, float, float, float, int]] = []
        for model, class_map in zip(self._models, self._per_model_class_map):
            predict_kwargs = dict(kwargs)
            if allowed_set is not None:
                submodel_allowed = sorted(
                    class_id for class_id, combined_id in class_map.items() if combined_id in allowed_set
                )
                if not submodel_allowed:
                    continue
                predict_kwargs["classes"] = submodel_allowed

            results = model.predict(source, **predict_kwargs)
            if not results:
                continue
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for index in range(len(boxes)):
                original_class_id = int(result.boxes.cls[index])
                combined_class_id = class_map.get(original_class_id)
                if combined_class_id is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in result.boxes.xyxy[index])
                confidence = float(result.boxes.conf[index])
                merged_detections.append((x1, y1, x2, y2, confidence, combined_class_id))

        merged_detections = _classwise_nms(merged_detections, float(kwargs.get("iou", 0.45)))
        return [_CombinedResult(merged_detections, self.names)]

    def __repr__(self) -> str:
        label = " + ".join(self._labels) if self._labels else f"{len(self._models)} models"
        return f"CombinedYOLOModel({label})"


def _load_world_model(model_path: str):
    fallback_path = os.environ.get("YOLO_FALLBACK_MODEL", "yolov8n.pt")
    if YOLOWorld is None:
        logger.warning("YOLOWorld unavailable. Falling back to %s.", fallback_path)
        return YOLO(fallback_path)
    try:
        model = YOLOWorld(model_path)
    except Exception:
        logger.warning("Could not load '%s'. Falling back to %s.", model_path, fallback_path)
        return YOLO(fallback_path)
    model.set_classes(sorted(INFERENCE_CLASS_NAMES))
    return model


def load_inference_model(model_path: str, base_model_path: str | None = None):
    """Load the live dashboard model, optionally merging base YOLO with trained weights."""
    if YOLO is None:
        raise RuntimeError("ultralytics is required unless --test is used.")

    resolved_path = str(model_path)
    if "world" in os.path.basename(resolved_path).lower():
        return _load_world_model(resolved_path)

    primary_model = YOLO(resolved_path)
    merge_base_path = _resolve_base_merge_path(resolved_path, base_model_path=base_model_path)
    if not merge_base_path:
        return primary_model

    base_model = YOLO(merge_base_path)
    logger.info(
        "Loaded combined live model using base '%s' plus trained '%s'",
        merge_base_path,
        resolved_path,
    )
    return CombinedYOLOModel(
        [base_model, primary_model],
        labels=[os.path.basename(merge_base_path), os.path.basename(resolved_path)],
    )
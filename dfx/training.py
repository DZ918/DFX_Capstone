"""Training pipeline: sample export, dataset management, and background training worker."""

import json
import os
import shutil
import threading
from datetime import datetime

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from dfx.constants import TRAIN_VIDEO_SAMPLE_MAX_FRAMES
from dfx.detection import safe_token
from dfx.alerts import read_alerts, write_alerts, ensure_alert_metadata
from dfx.gpu import get_best_device, is_jetson_linux, prepare_model_for_inference


def _move_model_to_device(model, device: str) -> None:
    """Best-effort device move for Ultralytics model wrappers and raw modules."""
    for candidate in (model, getattr(model, "model", None)):
        if candidate is None or not hasattr(candidate, "to"):
            continue
        candidate.to(device)


def _clear_cuda_cache() -> None:
    """Release cached CUDA allocations when torch is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


_original_pin_memory = None


def _disable_pin_memory_on_jetson() -> None:
    """On Jetson iGPU (unified memory), pin_memory is a no-op that wastes locked pages.
    Monkey-patch Tensor.pin_memory to return self so the dataloader skip pinning."""
    global _original_pin_memory
    if not is_jetson_linux() or _original_pin_memory is not None:
        return
    try:
        import torch
        _original_pin_memory = torch.Tensor.pin_memory
        torch.Tensor.pin_memory = lambda self, *a, **kw: self  # type: ignore[assignment]
    except Exception:
        pass


def _restore_pin_memory() -> None:
    """Undo the pin_memory patch."""
    global _original_pin_memory
    if _original_pin_memory is None:
        return
    try:
        import torch
        torch.Tensor.pin_memory = _original_pin_memory  # type: ignore[assignment]
    except Exception:
        pass
    _original_pin_memory = None


import glob as _glob


def _cleanup_after_training(config) -> None:
    """Remove snippet images/videos already exported to the dataset, and stale training run artifacts."""
    # 1. Delete exported snippet images (originals live in training_data/dataset/images now)
    snippet_dir = getattr(config, "snippet_dir", None)
    if snippet_dir and os.path.isdir(snippet_dir):
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
        for alert in alerts:
            status = str(alert.get("status", "")).strip().lower()
            if status not in ("accepted", "rejected"):
                continue
            for det in alert.get("detections", []):
                if not det.get("training_exported"):
                    continue
                snippet_file = str(det.get("snippet_file", "")).strip()
                if snippet_file:
                    path = os.path.join(snippet_dir, snippet_file)
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                    except OSError:
                        pass
            # Delete the source video if its frames were already exported
            if alert.get("training_video_exported"):
                video_file = str(alert.get("video_file", "")).strip()
                video_dir = getattr(config, "video_dir", None)
                if video_file and video_dir:
                    vpath = os.path.join(video_dir, video_file)
                    try:
                        if os.path.isfile(vpath):
                            os.remove(vpath)
                    except OSError:
                        pass

    # 2. Clean stale training run artifacts (plots, CSVs) but keep weights/
    runs_dir = getattr(config, "training_runs_dir", None)
    if runs_dir and os.path.isdir(runs_dir):
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            if not os.path.isdir(run_path):
                continue
            for item in os.listdir(run_path):
                item_path = os.path.join(run_path, item)
                if item == "weights":
                    continue  # keep trained weights
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                except OSError:
                    pass

    # 3. Remove stale labels.cache so it's rebuilt fresh next training
    cache_path = os.path.join(config.training_labels_dir, "..", "labels.cache")
    try:
        cache_path = os.path.normpath(cache_path)
        if os.path.isfile(cache_path):
            os.remove(cache_path)
    except OSError:
        pass


def _recommended_training_overrides(device: str) -> dict:
    """Use conservative training defaults on Jetson-class hardware."""
    overrides = {
        "workers": 2,
        "batch": 8,
    }
    if is_jetson_linux() and device.startswith("cuda"):
        overrides["workers"] = 1
        overrides["batch"] = 4
    return overrides


def read_class_map(path: str) -> dict[str, int]:
    """Load the class-name-to-index map used for accepted-sample training."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, dict):
                return {}
            parsed: dict[str, int] = {}
            for class_name, class_idx in data.items():
                if isinstance(class_name, str):
                    try:
                        parsed[class_name] = int(class_idx)
                    except (TypeError, ValueError):
                        continue
            return parsed
    except (OSError, json.JSONDecodeError):
        return {}


def write_class_map(path: str, class_map: dict[str, int]) -> None:
    """Persist the training class map in a deterministic format."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2, sort_keys=True)


def update_dataset_yaml(config, class_map: dict[str, int]) -> None:
    """Rewrite the YOLO dataset config so training reflects the current accepted classes."""
    os.makedirs(config.training_data_dir, exist_ok=True)
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    if not names:
        names = ["item"]
    yaml_lines = [
        f"path: {os.path.abspath(config.training_data_dir)}",
        "train: images",
        "val: images",
        "names:",
    ]
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name}")
    with open(config.training_yaml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(yaml_lines) + "\n")


def _normalized_xywh_from_xyxy(
    bbox_xyxy: list[float],
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> tuple[float, float, float, float] | None:
    """Project an XYXY box between frame sizes and return YOLO-normalized XYWH."""
    if source_width <= 1 or source_height <= 1 or target_width <= 1 or target_height <= 1:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y
    x1 = max(0.0, min(target_width - 1.0, x1))
    x2 = max(0.0, min(target_width, x2))
    y1 = max(0.0, min(target_height - 1.0, y1))
    y2 = max(0.0, min(target_height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) * 0.5) / target_width
    cy = ((y1 + y2) * 0.5) / target_height
    bw = (x2 - x1) / target_width
    bh = (y2 - y1) / target_height
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, bw)),
        max(0.0, min(1.0, bh)),
    )


def export_video_frames_for_training(alert: dict, config, class_map: dict[str, int]) -> int:
    """Extract labeled frames from one alert video so training can learn from motion clips."""
    if cv2 is None:
        return 0
    if not isinstance(alert, dict):
        return 0
    if alert.get("training_video_exported"):
        return 0
    video_file = str(alert.get("video_file", "")).strip()
    detections = alert.get("detections")
    frame_size = alert.get("frame_size")
    if not video_file or not isinstance(detections, list):
        return 0
    if not isinstance(frame_size, dict):
        return 0
    source_width = float(frame_size.get("width", 0) or 0)
    source_height = float(frame_size.get("height", 0) or 0)
    if source_width <= 1 or source_height <= 1:
        return 0
    if not config.video_dir:
        return 0
    video_path = os.path.join(config.video_dir, video_file)
    if not os.path.exists(video_path):
        return 0

    usable_detections: list[tuple[int, list[float], str]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        bbox_xyxy = det.get("bbox_xyxy")
        if (
            not class_name
            or not isinstance(bbox_xyxy, (list, tuple))
            or len(bbox_xyxy) != 4
        ):
            continue
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
        usable_detections.append((class_map[class_name], [float(v) for v in bbox_xyxy], class_name))
    if not usable_detections:
        return 0

    capture = cv2.VideoCapture(video_path)
    if not capture or not capture.isOpened():
        return 0
    exported = 0
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = TRAIN_VIDEO_SAMPLE_MAX_FRAMES
        sample_count = max(1, min(TRAIN_VIDEO_SAMPLE_MAX_FRAMES, total_frames))
        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = sorted({
                int(round(i * (max(0, total_frames - 1) / float(sample_count - 1))))
                for i in range(sample_count)
            })

        alert_id = str(alert.get("id", "alert")).strip() or "alert"
        for sample_idx, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or not hasattr(frame, "shape"):
                continue
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
            if frame_w <= 1 or frame_h <= 1:
                continue

            stem = f"{alert_id}_video_{sample_idx:02d}"
            image_name = f"{stem}.jpg"
            image_path = os.path.join(config.training_images_dir, image_name)
            label_path = os.path.join(config.training_labels_dir, f"{stem}.txt")
            if not cv2.imwrite(image_path, frame):
                continue

            label_lines: list[str] = []
            for class_id, bbox_xyxy, _class_name in usable_detections:
                normalized = _normalized_xywh_from_xyxy(
                    bbox_xyxy,
                    source_width=source_width,
                    source_height=source_height,
                    target_width=float(frame_w),
                    target_height=float(frame_h),
                )
                if normalized is None:
                    continue
                cx, cy, bw, bh = normalized
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not label_lines:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
                continue
            with open(label_path, "w", encoding="utf-8") as label_handle:
                label_handle.write("\n".join(label_lines) + "\n")
            exported += 1
    finally:
        capture.release()

    if exported > 0:
        alert["training_video_exported"] = True
        alert["training_video_samples"] = int(exported)
    return exported


def export_accepted_alert_samples(alert: dict, config) -> int:
    """Copy accepted snippets into a YOLO-style dataset and write matching label files."""
    if not isinstance(alert, dict):
        return 0
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return 0
    if not config.snippet_dir:
        return 0
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    class_map = read_class_map(config.class_map_path)
    accepted = 0
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        if det.get("training_exported"):
            continue
        snippet_file = str(det.get("snippet_file", "")).strip()
        class_name = str(det.get("class_name", "")).strip().lower()
        if not snippet_file or not class_name:
            continue
        source_path = os.path.join(config.snippet_dir, snippet_file)
        if not os.path.exists(source_path):
            continue
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
        class_id = class_map[class_name]
        ext = os.path.splitext(snippet_file)[1] or ".jpg"
        sample_stem = f"{alert.get('id', 'alert')}_{idx}_{safe_token(class_name)}"
        dest_image = os.path.join(config.training_images_dir, f"{sample_stem}{ext}")
        dest_label = os.path.join(config.training_labels_dir, f"{sample_stem}.txt")
        shutil.copy2(source_path, dest_image)
        snippet_bbox = det.get("snippet_bbox_xywhn")
        if not (
            isinstance(snippet_bbox, (list, tuple))
            and len(snippet_bbox) == 4
            and all(isinstance(v, (int, float)) for v in snippet_bbox)
        ):
            snippet_bbox = [0.5, 0.5, 1.0, 1.0]
        cx, cy, bw, bh = (max(0.0, min(1.0, float(v))) for v in snippet_bbox)
        with open(dest_label, "w", encoding="utf-8") as label_handle:
            label_handle.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        det["training_exported"] = True
        det["training_sample"] = os.path.basename(dest_image)
        accepted += 1
    accepted += export_video_frames_for_training(alert, config, class_map)
    write_class_map(config.class_map_path, class_map)
    update_dataset_yaml(config, class_map)
    return accepted


def export_rejected_alert_samples(alert: dict, config) -> int:
    """Copy rejected snippets into the dataset as negative samples with empty labels."""
    if not isinstance(alert, dict):
        return 0
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return 0
    if not config.snippet_dir:
        return 0
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    exported = 0
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        snippet_file = str(det.get("snippet_file", "")).strip()
        class_name = str(det.get("class_name", "")).strip().lower() or "item"
        if not snippet_file:
            continue
        source_path = os.path.join(config.snippet_dir, snippet_file)
        if not os.path.exists(source_path):
            continue
        ext = os.path.splitext(snippet_file)[1] or ".jpg"
        sample_stem = f"{alert.get('id', 'alert')}_{idx}_{safe_token(class_name)}_negative"
        dest_image = os.path.join(config.training_images_dir, f"{sample_stem}{ext}")
        dest_label = os.path.join(config.training_labels_dir, f"{sample_stem}.txt")
        shutil.copy2(source_path, dest_image)
        # Empty label files tell YOLO this image is a hard negative: it should contain none
        # of the tracked classes even though the detector previously thought it did.
        with open(dest_label, "w", encoding="utf-8") as label_handle:
            label_handle.write("")
        exported += 1
    update_dataset_yaml(config, read_class_map(config.class_map_path))
    return exported


def training_status_snapshot(config) -> dict:
    """Expose the last-known training state for the dashboard polling endpoint."""
    with config.training_lock:
        return {
            "running": bool(config.training_running),
            "last_started_at": config.training_last_started_at,
            "last_completed_at": config.training_last_completed_at,
            "last_error": config.training_last_error,
            "last_message": config.training_last_message,
            "last_weights": config.training_last_weights,
        }


def validate_training_environment() -> None:
    """Fail early with a readable message when binary dependencies are incompatible."""
    try:
        from matplotlib import font_manager  # noqa: F401
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        if "numpy.core.multiarray failed to import" in detail or "_ARRAY_API" in detail:
            raise RuntimeError(
                "Training dependencies are incompatible: reinstall or upgrade matplotlib so it matches the installed NumPy version."
            ) from exc
        raise RuntimeError(f"Training dependency import failed: {detail}") from exc


def _train_on_accepted_samples(config) -> None:
    """Background worker that exports accepted data, trains, and hot-swaps the model."""
    import time
    
    with config.training_lock:
        config.training_last_started_at = datetime.now().isoformat(timespec="seconds")
        config.training_last_error = ""
        config.training_last_message = "Preparing dataset"
        
    # [NEW] Suspend the live camera inference thread to free up Jetson Orin Nano VRAM
    was_detection_enabled = False
    with config.settings_lock:
        was_detection_enabled = config.detection_enabled
        config.detection_enabled = False
    
    inference_model = getattr(config, "model", None)
    inference_device = str(getattr(config, "inference_device", "cpu"))

    # Move the live inference model off GPU so training owns the limited Jetson VRAM.
    if inference_model is not None:
        with config.model_lock:
            try:
                _move_model_to_device(inference_model, "cpu")
            except Exception:
                pass
    _clear_cuda_cache()

    # Give the camera worker a moment to clear the VRAM pipeline.
    time.sleep(1.0)

    try:
        validate_training_environment()
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            changed = ensure_alert_metadata(alerts)
            exported_total = 0
            for alert in alerts:
                if str(alert.get("status", "")).strip().lower() == "accepted":
                    exported_total += export_accepted_alert_samples(alert, config)
            if changed or exported_total > 0:
                write_alerts(config.alert_log, alerts)
                
        image_files = [
            name for name in os.listdir(config.training_images_dir)
            if os.path.isfile(os.path.join(config.training_images_dir, name))
        ] if os.path.isdir(config.training_images_dir) else []
        
        if not image_files:
            raise RuntimeError("No accepted snippets available for training yet.")
        if YOLO is None:
            raise RuntimeError("ultralytics is required to train.")
            
        with config.training_lock:
            config.training_last_message = f"Training on {len(image_files)} accepted snippets"
        device = get_best_device()
        train_overrides = _recommended_training_overrides(device)

        # Jetson iGPU has no NVML support; use CUDA-native async allocator
        # to avoid NVML_SUCCESS assertion failures in CUDACachingAllocator.
        if is_jetson_linux():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

        # Disable pin_memory on Jetson unified memory (prevents OOM in dataloader)
        _disable_pin_memory_on_jetson()

        train_model = YOLO(config.model_path)
        train_result = train_model.train(
            data=config.training_yaml_path,
            epochs=config.train_epochs,
            imgsz=config.train_imgsz,
            project=config.training_runs_dir,
            name="accepted",
            exist_ok=True,
            verbose=False,
            device=device,
            batch=train_overrides["batch"],
            workers=train_overrides["workers"],
        )
        
        best_path = ""
        if hasattr(train_result, "save_dir"):
            candidate = os.path.join(str(train_result.save_dir), "weights", "best.pt")
            if os.path.exists(candidate):
                best_path = candidate
                
        with config.training_lock:
            config.training_last_completed_at = datetime.now().isoformat(timespec="seconds")
            config.training_last_weights = best_path
            config.training_last_message = "Training completed; weights saved for manual review"

        # Clean up exported snippets, videos, and stale run artifacts
        _cleanup_after_training(config)
            
    except Exception as exc:
        with config.training_lock:
            config.training_last_error = str(exc)
            config.training_last_message = "Training failed"
    finally:
        _restore_pin_memory()

        if inference_model is not None:
            with config.model_lock:
                try:
                    prepare_model_for_inference(inference_model, inference_device)
                except Exception:
                    pass
            _clear_cuda_cache()

        # [NEW] Restore the camera inference thread after training completes or fails
        with config.settings_lock:
            if was_detection_enabled:
                config.detection_enabled = True
                
        with config.training_lock:
            config.training_running = False


def start_training_job(config) -> bool:
    """Start the background training worker unless one is already running."""
    with config.training_lock:
        if config.training_running:
            return False
        config.training_running = True
        worker = threading.Thread(target=_train_on_accepted_samples, args=(config,), daemon=True)
        config.training_thread = worker
    worker.start()
    return True

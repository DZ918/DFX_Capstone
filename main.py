"""Minimal OpenCV + YOLO runner for image and webcam food/drink detection."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from uuid import uuid4

from pathlib import Path

from dfx.gpu import configure_jetson_gpu_env, predict_with_fallback, prepare_model_for_inference

configure_jetson_gpu_env()

try:
    import cv2
except Exception as exc:
    print(f"Warning: cv2 import failed: {exc}")
    cv2 = None

try:
    from ultralytics import YOLO, YOLOWorld
except Exception as exc:
    print(f"Warning: ultralytics import failed: {exc}")
    YOLO = None
    YOLOWorld = None


WRAPPER_CLASS_NAMES = {
    "food wrapper",
    "candy wrapper",
    "chocolate bar",
    "chocolate bar wrapper",
    "kit kat",
    "kitkat chocolate bar",
    "foil wrapped chocolate bar",
    "potato chip bag",
    "potato chips bag",
    "lays bag",
    "crisp packet",
    "snack bag",
    "crinkly plastic snack bag",
    "shiny snack packaging",
    "metalized snack bag",
    "plastic snack packaging",
    "granola bar wrapper",
    "gum package",
    "cellophane candy wrapper",
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
}

DETECTION_CLASS_NAMES = FOOD_CLASS_NAMES | WRAPPER_CLASS_NAMES
DEFAULT_WORLD_MODEL = "yolov8n.pt"


def _is_world_model(model_path: str) -> bool:
    return "world" in Path(model_path).name.lower()


def load_detection_model(model_path: str):
    """Load standard YOLO or YOLO-World with wrapper prompts."""
    if YOLO is None:
        raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")
    if _is_world_model(model_path):
        fallback_path = os.environ.get("YOLO_FALLBACK_MODEL", "yolov8n.pt")
        if YOLOWorld is None:
            print(
                "Warning: YOLOWorld is unavailable in this ultralytics build. "
                f"Falling back to {fallback_path}."
            )
            return YOLO(fallback_path)
        try:
            model = YOLOWorld(model_path)
        except Exception:
            print(
                f"Warning: could not load '{model_path}'. "
                f"Falling back to {fallback_path}."
            )
            return YOLO(fallback_path)
        model.set_classes(sorted(DETECTION_CLASS_NAMES))
        return model
    return YOLO(model_path)


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


def append_alert(log_path: str | None, alert: dict) -> None:
    """Append one alert record to the JSON log file, creating it if needed."""
    if not log_path:
        return
    alerts: list[dict] = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
                if isinstance(existing, list):
                    alerts = existing
        except (json.JSONDecodeError, OSError):
            alerts = []
    alerts.append(alert)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(alerts, handle, indent=2)


def _add_detection_snippets(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    alert_id: str,
) -> list[dict]:
    """Crop each detected object out of the frame and save it beside the alert."""
    if not snippet_dir:
        return detections
    os.makedirs(snippet_dir, exist_ok=True)
    height, width = frame.shape[:2]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        left = max(0, min(width - 1, int(x1)))
        top = max(0, min(height - 1, int(y1)))
        right = max(left + 1, min(width, int(x2)))
        bottom = max(top + 1, min(height, int(y2)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        class_token = safe_token(det.get("class_name", "item"))
        snippet_file = f"{alert_id}_{idx}_{class_token}.jpg"
        snippet_path = os.path.join(snippet_dir, snippet_file)
        if cv2.imwrite(snippet_path, crop):
            det["snippet_file"] = snippet_file
    return detections


def run_webcam(
    model,
    device: str,
    cam_index: int,
    conf: float,
    iou: float,
    persist_frames: int,
    cooldown: float,
    clear_frames: int,
    alert_log: str | None,
    snippet_dir: str | None,
) -> None:
    """Run the live webcam loop, annotate detections, and persist debounced alerts."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}.")

    try:
        allowed_ids = get_allowed_class_ids(model, DETECTION_CLASS_NAMES)
        consecutive = 0
        clear_count = 0
        armed = True
        last_alert_ts = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Some Ultralytics versions support a `classes` filter directly; older ones do not.
            results = predict_with_fallback(
                model,
                frame,
                verbose=False,
                conf=conf,
                iou=iou,
                classes=allowed_ids if allowed_ids else None,
                device=device,
            )
            result = results[0]
            detections = detections_from_result(result, allowed_names=DETECTION_CLASS_NAMES)
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
                label = f'{det["class_name"]} {det["confidence"]:.2f}'
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    2,
                )

            # The alert only fires after several consecutive detection frames and rearms only
            # after the scene has been clear long enough. That avoids duplicate alerts while
            # the same item remains in view.
            if detections:
                consecutive += 1
                clear_count = 0
            else:
                consecutive = 0
                clear_count += 1
                if clear_count >= max(1, clear_frames):
                    armed = True

            now = time.time()
            if (
                detections
                and consecutive >= max(1, persist_frames)
                and armed
                and (now - last_alert_ts) >= max(0.0, cooldown)
            ):
                alert_id = uuid4().hex[:12]
                alert = {
                    "id": alert_id,
                    "status": "new",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                    "detections": _add_detection_snippets(
                        frame,
                        detections,
                        snippet_dir=snippet_dir,
                        alert_id=alert_id,
                    ),
                }
                _append_alert(alert_log, alert)
                last_alert_ts = now
                armed = False

            cv2.imshow("YOLO + OpenCV (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(model, image_path: str, out_path: str | None, device: str) -> None:
    """Run detection once on a still image and either save or preview the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    allowed_ids = get_allowed_class_ids(model, DETECTION_CLASS_NAMES)
    results = predict_with_fallback(
        model,
        frame,
        verbose=False,
        classes=allowed_ids if allowed_ids else None,
        device=device,
    )
    detections = detections_from_result(results[0], allowed_names=DETECTION_CLASS_NAMES)
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )
    if out_path:
        cv2.imwrite(out_path, annotated)
        print(f"Wrote: {out_path}")
    else:
        cv2.imshow("YOLO + OpenCV (press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> int:
    """Parse CLI arguments, load the model, and dispatch to image or webcam mode."""
    parser = argparse.ArgumentParser(description="Quick OpenCV + YOLO demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_WORLD_MODEL,
        help="Path to model weights (YOLO-World recommended for wrapper detection)",
    )
    parser.add_argument("--image", help="Path to image for single-image demo")
    parser.add_argument("--out", help="Output path for annotated image")
    parser.add_argument("--cam", type=int, default=0, help="Camera index for webcam demo")
    parser.add_argument("--conf", type=float, default=0.12, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="IoU threshold")
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=5,
        help="Require this many consecutive frames with detections to alert",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=15.0,
        help="Minimum seconds between alerts",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=15,
        help="Require this many consecutive clear frames before re-arming alerts",
    )
    parser.add_argument(
        "--alert-log",
        default="alerts.json",
        help="Path to JSON alert log (set empty to disable)",
    )
    parser.add_argument(
        "--snippet-dir",
        default="snippets",
        help="Directory where per-detection crop images are stored (set empty to disable)",
    )
    args = parser.parse_args()
    if cv2 is None:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")

    model = load_detection_model(args.model)
    inference_device = prepare_model_for_inference(model)
    print(f"Using inference device: {inference_device}")

    if args.image:
        run_image(model, args.image, args.out, inference_device)
    else:
        alert_log = args.alert_log or None
        run_webcam(
            model,
            inference_device,
            args.cam,
            conf=args.conf,
            iou=args.iou,
            persist_frames=args.persist_frames,
            cooldown=args.cooldown,
            clear_frames=args.clear_frames,
            alert_log=alert_log,
            snippet_dir=args.snippet_dir or None,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

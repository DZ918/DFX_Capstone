import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2

try:
    from ultralytics import YOLO
except Exception as exc:
    print("Missing dependency.")
    raise


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


def get_allowed_class_ids(model, allowed_names: set[str]) -> list[int]:
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


def detections_from_result(result, allowed_names: set[str] | None = None) -> list[dict]:
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


def run_webcam(
    model,
    cam_index: int,
    conf: float,
    iou: float,
    persist_frames: int,
    cooldown: float,
    clear_frames: int,
    alert_log: str | None,
) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}.")

    try:
        allowed_ids = get_allowed_class_ids(model, FOOD_CLASS_NAMES)
        consecutive = 0
        clear_count = 0
        armed = True
        last_alert_ts = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                results = model.predict(
                    frame,
                    verbose=False,
                    conf=conf,
                    iou=iou,
                    classes=allowed_ids if allowed_ids else None,
                )
            except TypeError:
                results = model.predict(frame, verbose=False, conf=conf, iou=iou)
            result = results[0]
            detections = detections_from_result(result, allowed_names=FOOD_CLASS_NAMES)
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
                alert = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                    "detections": detections,
                }
                append_alert(alert_log, alert)
                last_alert_ts = now
                armed = False

            cv2.imshow("YOLO + OpenCV (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(model, image_path: str, out_path: str | None) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    allowed_ids = get_allowed_class_ids(model, FOOD_CLASS_NAMES)
    try:
        results = model.predict(
            frame, verbose=False, classes=allowed_ids if allowed_ids else None
        )
    except TypeError:
        results = model.predict(frame, verbose=False)
    detections = detections_from_result(results[0], allowed_names=FOOD_CLASS_NAMES)
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
    parser = argparse.ArgumentParser(description="Quick OpenCV + YOLO demo")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--image", help="Path to image for single-image demo")
    parser.add_argument("--out", help="Output path for annotated image")
    parser.add_argument("--cam", type=int, default=0, help="Camera index for webcam demo")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=3,
        help="Require this many consecutive frames with detections to alert",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="Minimum seconds between alerts",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=10,
        help="Require this many consecutive clear frames before re-arming alerts",
    )
    parser.add_argument(
        "--alert-log",
        default="alerts.json",
        help="Path to JSON alert log (set empty to disable)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        run_image(model, args.image, args.out)
    else:
        alert_log = args.alert_log or None
        run_webcam(
            model,
            args.cam,
            conf=args.conf,
            iou=args.iou,
            persist_frames=args.persist_frames,
            cooldown=args.cooldown,
            clear_frames=args.clear_frames,
            alert_log=alert_log,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

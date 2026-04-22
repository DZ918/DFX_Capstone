"""Browser dashboard for live detection, alert review, and accepted-sample training."""

import argparse
import glob
import os
import queue
import sys
import threading
import time
from http.server import ThreadingHTTPServer

# Import the modular architecture to prevent code duplication
from dfx.gpu import (
    configure_jetson_gpu_env,
    get_best_device,
    is_jetson_linux,
    predict_with_fallback,
    prepare_model_for_inference,
)
from dfx.camera import camera_worker
from dfx.alerts import append_alert, create_alert
from dfx.multicam import CameraManager
from dfx.server import DashboardHandler, HTML_PAGE as DASHBOARD_HTML_PAGE
from dfx.settings import DashboardConfig
from dfx.constants import DEFAULT_MAP_IMAGE_PATH, INFERENCE_CLASS_NAMES

try:
    from ultralytics import YOLO, YOLOWorld
except Exception:
    YOLO = None
    YOLOWorld = None

# Ensure Jetson Orin Nano CUDA libraries are linked before PyTorch initializes
configure_jetson_gpu_env()

def load_inference_model(model_path: str):
    """Load YOLO model; configure open-vocabulary classes for YOLO-World."""
    if YOLO is None:
        raise RuntimeError("ultralytics is required unless --test is used.")
    if "world" in os.path.basename(str(model_path)).lower():
        fallback_path = os.environ.get("YOLO_FALLBACK_MODEL", "yolov8n.pt")
        if YOLOWorld is None:
            print(f"Warning: YOLOWorld unavailable. Falling back to {fallback_path}.")
            return YOLO(fallback_path)
        try:
            model = YOLOWorld(model_path)
        except Exception:
            print(f"Warning: could not load '{model_path}'. Falling back to {fallback_path}.")
            return YOLO(fallback_path)
        model.set_classes(sorted(INFERENCE_CLASS_NAMES))
        return model
    return YOLO(model_path)


class StandaloneDashboardHandler(DashboardHandler):
    """Overrides the base handler to serve the inline HTML page directly."""
    def do_GET(self):
        if self.path == "/":
            self._send_html(DASHBOARD_HTML_PAGE)
            return
        # Route all other API endpoints to the shared dfx/server.py logic
        super().do_GET()


def _frame_relay_worker(config):
    """Move encoded frames from the producer queue into the shared latest-frame snapshot."""
    while not config.stop:
        frame_queue = getattr(config, "frame_queue", None)
        if frame_queue is None:
            time.sleep(0.05)
            continue
        try:
            frame, jpeg_payload, frame_ts = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        config.latest_frame = frame
        config.latest_jpeg = jpeg_payload
        config.latest_frame_ts = float(frame_ts)
        frame_ready_event = getattr(config, "frame_ready_event", None)
        if frame_ready_event is not None:
            frame_ready_event.set()


def _alert_persist_worker(config):
    """Persist queued alert jobs without blocking the realtime camera loop."""
    while True:
        alert_queue = getattr(config, "alert_queue", None)
        if alert_queue is None:
            if config.stop:
                return
            time.sleep(0.05)
            continue
        if config.stop and alert_queue.empty():
            return
        try:
            payload = alert_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            alert = create_alert(
                payload["frame"],
                payload["detections"],
                snippet_dir=payload["snippet_dir"],
                video_dir=payload["video_dir"],
                recent_frames=payload["recent_frames"],
                video_fps=payload["video_fps"],
                camera_zone=payload["camera_zone"],
                context_detections=payload["context_detections"],
                motion_detected=payload["motion_detected"],
                motion_score=payload["motion_score"],
                hand_to_mouth_source=payload["hand_to_mouth_source"],
                hand_to_mouth_event_count=payload["hand_to_mouth_event_count"],
                attach_video=payload["attach_video"],
                alert_reason=payload["alert_reason"],
            )
            with config.alert_lock:
                append_alert(
                    config.alert_log,
                    alert,
                    summary_csv_path=config.detection_summary_csv,
                )
        except Exception as exc:
            config.alert_worker_last_error = str(exc)
            print(f"Warning: failed to persist alert job: {exc}")
        finally:
            alert_queue.task_done()


def main():
    """Start the camera worker and HTTP server that power the dashboard."""
    parser = argparse.ArgumentParser(description="Camera dashboard with live alerts")
    parser.add_argument("--test", action="store_true", help="Run with synthetic feed/alerts")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--alert-log", default="alerts.json", help="Alert JSON path")
    parser.add_argument("--detection-summary-csv", default="detections_summary.csv")
    parser.add_argument("--snippet-dir", default="snippets")
    parser.add_argument("--training-dir", default="training_data")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--inference-imgsz", type=int, default=640)
    parser.add_argument("--max-inference-fps", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--motion-hold-seconds", type=float, default=0.1)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-zone", default="Zone A")
    parser.add_argument("--map-image", default=DEFAULT_MAP_IMAGE_PATH)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--persist-frames", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--clear-frames", type=int, default=15)
    parser.add_argument("--motion-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-window", type=int, default=12)
    parser.add_argument("--motion-displacement-threshold", type=float, default=0.07)
    parser.add_argument("--motion-upward-threshold", type=float, default=0.02)
    args = parser.parse_args()

    model = None if args.test else load_inference_model(args.model)
    inference_device = "cpu" if args.test else prepare_model_for_inference(model)
    
    if not args.test:
        print(f"Hardware Check: Bound to {inference_device} on Jetson Orin Nano")

    config = DashboardConfig(
        model=model,
        model_path=args.model,
        alert_log=args.alert_log,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        conf=args.conf,
        iou=args.iou,
        persist_frames=args.persist_frames,
        cooldown=args.cooldown,
        clear_frames=args.clear_frames,
        camera_zone=args.camera_zone,
        map_image_path=args.map_image,
        snippet_dir=args.snippet_dir or None,
        detection_summary_csv=args.detection_summary_csv or None,
        inference_imgsz=args.inference_imgsz,
        max_inference_fps=args.max_inference_fps,
        jpeg_quality=args.jpeg_quality,
        motion_hold_seconds=args.motion_hold_seconds,
        training_dir=args.training_dir,
        train_epochs=args.train_epochs,
        train_imgsz=args.train_imgsz,
        motion_enabled=args.motion_enabled,
        motion_window=args.motion_window,
        motion_displacement_threshold=args.motion_displacement_threshold,
        motion_upward_threshold=args.motion_upward_threshold,
        test_mode=args.test,
    )
    config.inference_device = inference_device
    config.frame_queue = queue.Queue(maxsize=2)
    config.frame_ready_event = threading.Event()
    config.latest_frame_ts = 0.0
    config.alert_queue = queue.Queue(maxsize=3)
    config.alert_worker_last_error = ""
    config.alert_jobs_dropped = 0

    frame_relay_worker = threading.Thread(target=_frame_relay_worker, args=(config,), daemon=True)
    frame_relay_worker.start()
    alert_persist_worker = threading.Thread(target=_alert_persist_worker, args=(config,), daemon=True)
    alert_persist_worker.start()

    if not args.test:
        worker = threading.Thread(target=camera_worker, args=(config, args.cam), daemon=True)
        worker.start()

    camera_manager = CameraManager(
        primary_camera_index=config.camera_index,
        stream_fps=config.stream_fps,
        width=config.width,
        height=config.height,
        jpeg_quality=config.jpeg_quality,
    )
    server = ThreadingHTTPServer((args.host, args.port), StandaloneDashboardHandler)
    server.config = config
    server.camera_manager = camera_manager
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        config.stop = True
        camera_manager.shutdown()
        config.frame_ready_event.set()

if __name__ == "__main__":
    sys.exit(main())
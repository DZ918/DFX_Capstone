"""Browser dashboard for live detection, alert review, and accepted-sample training."""

import argparse
import glob
import os
import sys
import threading
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

    if not args.test:
        worker = threading.Thread(target=camera_worker, args=(config, args.cam), daemon=True)
        worker.start()

    server = ThreadingHTTPServer((args.host, args.port), StandaloneDashboardHandler)
    server.config = config
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        config.stop = True

if __name__ == "__main__":
    sys.exit(main())
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
from dfx.model_loader import load_inference_model
from dfx.server import DashboardHandler, HTML_PAGE as DASHBOARD_HTML_PAGE
from dfx.settings import DashboardConfig
from dfx.training import refresh_runtime_class_names
from dfx.constants import (
    DASHBOARD_PRIMARY_CAMERA_INDEX,
    DASHBOARD_PRIMARY_CAMERA_LABEL,
    DASHBOARD_PRIMARY_CAMERA_ZONE,
    DASHBOARD_SECONDARY_CAMERA_INDEX,
    DASHBOARD_SECONDARY_CAMERA_LABEL,
    DASHBOARD_SECONDARY_CAMERA_ZONE,
    DEFAULT_DASHBOARD_CONFIDENCE,
    DEFAULT_DASHBOARD_MODEL_PATH,
    DEFAULT_MAP_IMAGE_PATH,
    INFERENCE_CLASS_NAMES,
)

# Ensure Jetson Orin Nano CUDA libraries are linked before PyTorch initializes
configure_jetson_gpu_env()


FIXED_PRIMARY_CAMERA_INDEX = int(DASHBOARD_PRIMARY_CAMERA_INDEX)
FIXED_SECONDARY_CAMERA_INDEX = int(DASHBOARD_SECONDARY_CAMERA_INDEX)
FIXED_PRIMARY_CAMERA_LABEL = str(DASHBOARD_PRIMARY_CAMERA_LABEL)
FIXED_SECONDARY_CAMERA_LABEL = str(DASHBOARD_SECONDARY_CAMERA_LABEL)
FIXED_PRIMARY_CAMERA_ZONE = str(DASHBOARD_PRIMARY_CAMERA_ZONE)
FIXED_SECONDARY_CAMERA_ZONE = str(DASHBOARD_SECONDARY_CAMERA_ZONE)


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
                video_required=payload.get("video_required", False),
                prefer_mp4=payload.get("prefer_mp4", False),
                alert_reason=payload["alert_reason"],
            )
            requires_strict_proxy_recording = bool(
                payload.get("video_required", False)
                and str(payload.get("alert_reason", "")).strip().lower() == "motion_burst"
                and str(payload.get("hand_to_mouth_source", "")).strip().lower() == "person_proxy"
            )
            if requires_strict_proxy_recording:
                if alert is None:
                    print(
                        "Warning: dropping proxy hand-to-mouth alert because required buffered recording was not created"
                    )
                    continue
                video_file = str(alert.get("video_file", "")).strip()
                if not video_file:
                    snippet_dir = str(payload.get("snippet_dir", "")).strip()
                    detections = alert.get("detections") if isinstance(alert, dict) else None
                    if snippet_dir and isinstance(detections, list):
                        snippet_root = os.path.abspath(snippet_dir)
                        for det in detections:
                            if not isinstance(det, dict):
                                continue
                            snippet_file = os.path.basename(str(det.get("snippet_file", "")).strip())
                            if not snippet_file or snippet_file in {".", ".."}:
                                continue
                            snippet_candidate = os.path.abspath(os.path.join(snippet_root, snippet_file))
                            try:
                                if os.path.commonpath([snippet_root, snippet_candidate]) == snippet_root:
                                    if os.path.exists(snippet_candidate):
                                        os.remove(snippet_candidate)
                            except (OSError, ValueError):
                                pass
                    print(
                        "Warning: dropping proxy hand-to-mouth alert because generated recording filename is missing"
                    )
                    continue
                video_dir = str(payload.get("video_dir", "")).strip()
                safe_video_path = ""
                if video_dir and video_file:
                    safe_video_path = os.path.abspath(os.path.join(video_dir, os.path.basename(video_file)))
                    try:
                        if os.path.commonpath([os.path.abspath(video_dir), safe_video_path]) != os.path.abspath(video_dir):
                            safe_video_path = ""
                    except ValueError:
                        safe_video_path = ""
                video_size = -1
                if safe_video_path and os.path.exists(safe_video_path):
                    try:
                        video_size = int(os.path.getsize(safe_video_path))
                    except OSError:
                        video_size = -1
                if video_size <= 0:
                    if safe_video_path:
                        try:
                            os.remove(safe_video_path)
                        except OSError:
                            pass
                    print(
                        "Warning: dropping proxy hand-to-mouth alert because buffered recording is empty or missing"
                    )
                    continue
            if alert is None:
                continue
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


def _camera_worker_supervisor(config):
    """Keep the primary camera worker alive and auto-restart it after unexpected crashes."""
    while not config.stop:
        try:
            camera_worker(config, config.camera_index)
        except Exception as exc:
            print(f"Warning: camera worker crashed; restarting in 0.5s: {exc}")
            if config.stop:
                break
            time.sleep(0.5)
            continue
        # camera_worker normally runs until config.stop becomes True.
        if config.stop:
            break
        time.sleep(0.1)


def main():
    """Start the camera worker and HTTP server that power the dashboard."""
    parser = argparse.ArgumentParser(description="Camera dashboard with live alerts")
    parser.add_argument("--test", action="store_true", help="Run with synthetic feed/alerts")
    parser.add_argument("--model", default=DEFAULT_DASHBOARD_MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument("--cam", type=int, default=FIXED_PRIMARY_CAMERA_INDEX, help="Camera index")
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
    parser.add_argument("--inference-imgsz", type=int, default=960)
    parser.add_argument("--max-inference-fps", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--motion-hold-seconds", type=float, default=0.1)
    parser.add_argument("--camera-index", type=int, default=FIXED_PRIMARY_CAMERA_INDEX)
    parser.add_argument("--camera-zone", default=FIXED_PRIMARY_CAMERA_ZONE)
    parser.add_argument("--map-image", default=DEFAULT_MAP_IMAGE_PATH)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--conf", type=float, default=DEFAULT_DASHBOARD_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--persist-frames", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--clear-frames", type=int, default=15)
    parser.add_argument("--motion-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-window", type=int, default=12)
    parser.add_argument("--motion-displacement-threshold", type=float, default=0.07)
    parser.add_argument("--motion-upward-threshold", type=float, default=0.02)
    args = parser.parse_args()

    # The dashboard uses a fixed two-camera layout: Camera 1 -> Zone G, Camera 2 -> Zone F.
    args.cam = FIXED_PRIMARY_CAMERA_INDEX
    args.camera_index = FIXED_PRIMARY_CAMERA_INDEX
    args.camera_zone = FIXED_PRIMARY_CAMERA_ZONE

    model = None if args.test else load_inference_model(args.model)
    inference_device = "cpu" if args.test else prepare_model_for_inference(model)
    
    if not args.test:
        print(f"Hardware Check: Bound to {inference_device} on Jetson Orin Nano")
        print(
            "Camera Layout: "
            f"{FIXED_PRIMARY_CAMERA_LABEL} -> index {FIXED_PRIMARY_CAMERA_INDEX} ({FIXED_PRIMARY_CAMERA_ZONE}), "
            f"{FIXED_SECONDARY_CAMERA_LABEL} -> index {FIXED_SECONDARY_CAMERA_INDEX} ({FIXED_SECONDARY_CAMERA_ZONE})"
        )

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
    runtime_food_names, runtime_inference_names = refresh_runtime_class_names(config)
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
        worker = threading.Thread(target=_camera_worker_supervisor, args=(config,), daemon=True)
        worker.start()

    camera_manager = CameraManager(
        primary_camera_index=config.camera_index,
        stream_fps=config.stream_fps,
        width=config.width,
        height=config.height,
        jpeg_quality=config.jpeg_quality,
        model=config.model,
        model_lock=config.model_lock,
        detection_enabled=config.detection_enabled,
        conf=config.conf,
        iou=config.iou,
        inference_imgsz=config.inference_imgsz,
        max_inference_fps=config.max_inference_fps,
        inference_device=config.inference_device,
        tracked_class_names=runtime_food_names,
        allowed_class_names=runtime_inference_names,
    )
    config.camera_manager = camera_manager
    secondary_start = camera_manager.add_camera(FIXED_SECONDARY_CAMERA_INDEX)
    if not secondary_start.get("ok"):
        print(
            "Warning: secondary camera startup failed "
            f"(index {FIXED_SECONDARY_CAMERA_INDEX}): {secondary_start.get('error', 'unknown error')}"
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
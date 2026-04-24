"""HTTP request handler serving the dashboard page, JSON APIs, and MJPEG stream."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from dfx.alerts import (
    append_alert,
    build_consumption_stats,
    clamp_box,
    ensure_alert_metadata,
    read_alerts,
    write_alerts,
)
from dfx.camera import list_camera_devices
from dfx.detection import make_placeholder_svg, make_random_alerts
from dfx.settings import settings_snapshot, update_runtime_settings, reset_runtime_settings
from dfx.training import (
    export_accepted_alert_samples,
    export_rejected_alert_samples,
    read_class_map,
    start_training_job,
    training_status_snapshot,
)
from dfx.constants import FOOD_CLASS_NAMES

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")


def _load_html_page() -> str:
    path = os.path.join(_TEMPLATE_DIR, "dashboard.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


HTML_PAGE = _load_html_page()


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard page, JSON APIs, snippet images, and MJPEG stream."""
    server_version = "FoodDrinkDashboard/0.2"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/alerts":
            params = parse_qs(parsed.query)
            try:
                limit = int(params.get("limit", [50])[0])
            except ValueError:
                limit = 50
            limit = max(0, min(500, limit))
            self._send_alerts(limit)
            return
        if parsed.path == "/settings":
            self._send_settings()
            return
        if parsed.path == "/cameras/active":
            self._send_active_cameras()
            return
        if parsed.path == "/cameras":
            self._send_cameras()
            return
        if parsed.path == "/train/status":
            self._send_train_status()
            return
        if parsed.path == "/stats/consumption":
            self._send_consumption_stats()
            return
        if parsed.path == "/map-image":
            self._send_map_image()
            return
        if parsed.path == "/snapshot":
            params = parse_qs(parsed.query)
            camera_values = params.get("camera", [])
            self._send_snapshot(camera_values[0] if camera_values else None)
            return
        if parsed.path == "/classes":
            self._send_classes()
            return
        if parsed.path.startswith("/snippets/"):
            self._send_snippet(parsed.path.removeprefix("/snippets/"))
            return
        if parsed.path.startswith("/videos/"):
            self._send_video(parsed.path.removeprefix("/videos/"))
            return
        if parsed.path == "/stream":
            params = parse_qs(parsed.query)
            camera_values = params.get("camera", [])
            if camera_values:
                self._stream_preview_mjpeg(camera_values[0])
                return
            self._stream_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/cameras/add":
            self._add_active_camera()
            return
        if parsed.path == "/cameras/remove":
            self._remove_active_camera()
            return
        if parsed.path == "/alerts/manage":
            self._manage_alert()
            return
        if parsed.path == "/settings/reset":
            self._reset_settings()
            return
        if parsed.path == "/settings":
            self._update_settings()
            return
        if parsed.path == "/train/accepted":
            self._trigger_train_accepted()
            return
        if parsed.path == "/annotate":
            self._create_manual_annotation()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, payload: dict | list, status=HTTPStatus.OK):
        """Send a JSON response with no-cache headers for live dashboard polling."""
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        """Send the inline dashboard HTML page."""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        """Read and parse a bounded JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            raise ValueError("Missing request body")
        raw = self.rfile.read(min(content_length, 1_000_000))
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("Request body must be valid JSON") from None

    def _send_alerts(self, limit):
        """Return recent alerts, using generated data in test mode."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            frame_width = config.width or 640
            frame_height = config.height or 360
            alerts = make_random_alerts(limit, frame_width, frame_height)
            self._send_json(alerts, HTTPStatus.OK)
            return
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            if ensure_alert_metadata(alerts):
                write_alerts(config.alert_log, alerts)
        if limit > 0:
            alerts = alerts[-limit:]
        self._send_json(alerts, HTTPStatus.OK)

    def _manage_alert(self):
        """Accept, reject, or delete an alert and persist the updated alert log."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            self.send_error(HTTPStatus.BAD_REQUEST, "Alert management disabled in --test mode")
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        alert_id = str(payload.get("alert_id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        if not alert_id or action not in {"accept", "reject", "delete"}:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid alert_id or action")
            return

        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            ensure_alert_metadata(alerts)
            target_index = -1
            for idx, alert in enumerate(alerts):
                if isinstance(alert, dict) and alert.get("id") == alert_id:
                    target_index = idx
                    break
            if target_index < 0:
                self.send_error(HTTPStatus.NOT_FOUND, "Alert not found")
                return
            if action == "reject":
                export_rejected_alert_samples(alerts[target_index], config)
                alerts.pop(target_index)
            elif action == "delete":
                alerts.pop(target_index)
            else:
                # Accepting an alert also exports its snippets into the training dataset.
                exported_count = export_accepted_alert_samples(alerts[target_index], config)
                alerts[target_index]["status"] = "accepted"
                if exported_count > 0:
                    try:
                        current_samples = int(alerts[target_index].get("accepted_samples", 0))
                    except (TypeError, ValueError):
                        current_samples = 0
                    alerts[target_index]["accepted_samples"] = current_samples + exported_count
                alerts[target_index]["accepted_at"] = datetime.now().isoformat(timespec="seconds")
                alerts[target_index]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            write_alerts(config.alert_log, alerts)
        self._send_json({"ok": True, "action": action, "alert_id": alert_id}, HTTPStatus.OK)

    def _send_settings(self):
        """Return the current runtime settings to the browser."""
        config: DashboardConfig = self.server.config
        self._send_json(settings_snapshot(config), HTTPStatus.OK)

    def _send_cameras(self):
        """Return the currently probeable webcam devices for the dropdown."""
        self._send_json(list_camera_devices(), HTTPStatus.OK)

    def _camera_manager(self):
        """Return the optional auxiliary camera preview manager attached to the server."""
        return getattr(self.server, "camera_manager", None)

    def _resolve_camera_source(self, camera_index_value=None):
        """Resolve the requested camera into its current frame source and UI label."""
        config = self.server.config
        with config.settings_lock:
            primary_index = int(config.camera_index)
            primary_zone = str(config.camera_zone)

        if camera_index_value in {None, "", primary_index}:
            return {
                "camera_index": primary_index,
                "zone": primary_zone,
                "frame": config.latest_frame,
                "jpeg": config.latest_jpeg,
            }

        try:
            requested_index = int(camera_index_value)
        except (TypeError, ValueError):
            raise ValueError("Invalid camera index") from None

        if requested_index == primary_index:
            return {
                "camera_index": primary_index,
                "zone": primary_zone,
                "frame": config.latest_frame,
                "jpeg": config.latest_jpeg,
            }

        camera_manager = self._camera_manager()
        if camera_manager is None:
            raise LookupError("Multi-camera manager unavailable")
        preview = camera_manager.get_camera(requested_index)
        if preview is None:
            raise LookupError(f"Camera {requested_index} is not active")
        return {
            "camera_index": requested_index,
            "zone": f"Camera {requested_index}",
            "frame": preview.get_frame(),
            "jpeg": preview.get_jpeg(),
        }

    def _sync_camera_manager(self, config):
        """Keep the preview manager aligned with the current primary camera settings."""
        camera_manager = self._camera_manager()
        if camera_manager is None:
            return
        with config.settings_lock:
            camera_index = int(config.camera_index)
            stream_fps = float(config.stream_fps)
            width = int(config.width)
            height = int(config.height)
            jpeg_quality = int(config.jpeg_quality)
            detection_enabled = bool(config.detection_enabled)
            conf = float(config.conf)
            iou = float(config.iou)
            inference_imgsz = int(config.inference_imgsz)
            max_inference_fps = float(config.max_inference_fps)
            inference_device = str(getattr(config, "inference_device", "cpu"))
        camera_manager.update_primary_camera(camera_index)
        camera_manager.update_stream_settings(
            stream_fps=stream_fps,
            width=width,
            height=height,
            jpeg_quality=jpeg_quality,
            detection_enabled=detection_enabled,
            conf=conf,
            iou=iou,
            inference_imgsz=inference_imgsz,
            max_inference_fps=max_inference_fps,
            inference_device=inference_device,
        )

    def _send_active_cameras(self):
        """Return the primary camera plus any auxiliary preview cameras shown in the grid."""
        config = self.server.config
        self._sync_camera_manager(config)
        with config.settings_lock:
            primary_index = int(config.camera_index)
            camera_enabled = bool(config.camera_enabled)
        primary_available = bool(getattr(config, "camera_available", camera_enabled))
        primary_error = str(getattr(config, "camera_error", ""))
        if not camera_enabled:
            primary_available = False
            if not primary_error:
                primary_error = "Camera is OFF"
        cameras = [
            {
                "index": primary_index,
                "role": "primary",
                "available": primary_available,
                "error": primary_error,
                "stream_url": "/stream",
            }
        ]
        camera_manager = self._camera_manager()
        if camera_manager is not None:
            for preview in camera_manager.status_snapshot():
                cameras.append(
                    {
                        **preview,
                        "role": "preview",
                        "stream_url": f"/stream?camera={int(preview['index'])}",
                    }
                )
        self._send_json(
            {
                "primary_camera_index": primary_index,
                "cameras": cameras,
            },
            HTTPStatus.OK,
        )

    def _add_active_camera(self):
        """Start one auxiliary preview camera without restarting the dashboard."""
        camera_manager = self._camera_manager()
        if camera_manager is None:
            self._send_json({"ok": False, "error": "Multi-camera manager unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        result = camera_manager.add_camera(payload.get("index"))
        if not result.get("ok"):
            error = str(result.get("error", "Could not add camera"))
            status = HTTPStatus.CONFLICT if "already" in error.lower() else HTTPStatus.BAD_REQUEST
            self._send_json(result, status)
            return
        self._send_json(result, HTTPStatus.OK)

    def _remove_active_camera(self):
        """Stop one auxiliary preview camera without restarting the dashboard."""
        camera_manager = self._camera_manager()
        if camera_manager is None:
            self._send_json({"ok": False, "error": "Multi-camera manager unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        result = camera_manager.remove_camera(payload.get("index"))
        if not result.get("ok"):
            self._send_json(result, HTTPStatus.NOT_FOUND)
            return
        self._send_json(result, HTTPStatus.OK)

    def _update_settings(self):
        """Apply settings posted from the dashboard form."""
        config: DashboardConfig = self.server.config
        try:
            payload = self._read_json_body()
            updated = update_runtime_settings(config, payload)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._sync_camera_manager(config)
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _reset_settings(self):
        """Restore runtime settings to the startup defaults."""
        config: DashboardConfig = self.server.config
        updated = reset_runtime_settings(config)
        self._sync_camera_manager(config)
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _send_train_status(self):
        """Return the latest background-training status snapshot."""
        config: DashboardConfig = self.server.config
        self._send_json(training_status_snapshot(config), HTTPStatus.OK)

    def _send_consumption_stats(self):
        """Return aggregate eating/drinking counts for the dashboard table."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            frame_width = config.width or 640
            frame_height = config.height or 360
            alerts = make_random_alerts(40, frame_width, frame_height)
            self._send_json(build_consumption_stats(alerts), HTTPStatus.OK)
            return
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            if ensure_alert_metadata(alerts):
                write_alerts(config.alert_log, alerts)
        self._send_json(build_consumption_stats(alerts), HTTPStatus.OK)

    def _trigger_train_accepted(self):
        """Start training on accepted snippets if the environment supports it."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            self.send_error(HTTPStatus.BAD_REQUEST, "Training is disabled in --test mode")
            return
        if YOLO is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "ultralytics is required to train")
            return
        started = start_training_job(config)
        if not started:
            self.send_error(HTTPStatus.CONFLICT, "Training is already running")
            return
        self._send_json({"ok": True, "started": True}, HTTPStatus.ACCEPTED)

    def _send_snippet(self, encoded_name: str):
        """Serve one saved detection crop after validating the requested filename."""
        config: DashboardConfig = self.server.config
        if not config.snippet_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Snippet storage is disabled")
            return
        requested_name = unquote(encoded_name)
        if requested_name != os.path.basename(requested_name):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid snippet path")
            return
        snippet_root = os.path.abspath(config.snippet_dir)
        snippet_path = os.path.abspath(os.path.join(snippet_root, requested_name))
        if not snippet_path.startswith(f"{snippet_root}{os.sep}"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid snippet path")
            return
        if not os.path.exists(snippet_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Snippet not found")
            return
        with open(snippet_path, "rb") as handle:
            body = handle.read()
        content_type = "image/jpeg"
        lower = snippet_path.lower()
        if lower.endswith(".png"):
            content_type = "image/png"
        elif lower.endswith(".webp"):
            content_type = "image/webp"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_map_image(self):
        """Serve the configured DFX lab layout image used by the map modal."""
        config: DashboardConfig = self.server.config
        map_path = os.path.abspath(str(config.map_image_path))
        if not os.path.exists(map_path):
            # Fallback for Linux/case-sensitive filesystems and alternate extensions.
            root_dir = os.path.dirname(map_path)
            candidates = [
                "dfx_lab_map.png",
                "dfx_lab_map.jpg",
                "dfx_lab_map.jpeg",
                "dfx_lab_map.webp",
                "dfx_lab_map.svg",
            ]
            for candidate in candidates:
                probe = os.path.join(root_dir, candidate)
                if os.path.exists(probe):
                    map_path = probe
                    break
        if not os.path.exists(map_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Map image not found")
            return
        with open(map_path, "rb") as handle:
            body = handle.read()
        content_type = "image/png"
        lower = map_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif lower.endswith(".webp"):
            content_type = "image/webp"
        elif lower.endswith(".svg"):
            content_type = "image/svg+xml"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self, encoded_name: str):
        """Serve one saved alert video clip after validating the filename."""
        config: DashboardConfig = self.server.config
        if not config.video_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Video storage is disabled")
            return
        requested_name = unquote(encoded_name)
        if requested_name != os.path.basename(requested_name):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video path")
            return
        video_root = os.path.abspath(config.video_dir)
        video_path = os.path.abspath(os.path.join(video_root, requested_name))
        if not video_path.startswith(f"{video_root}{os.sep}"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video path")
            return
        if not os.path.exists(video_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        content_type = "video/mp4"
        lower = video_path.lower()
        if lower.endswith(".avi"):
            content_type = "video/x-msvideo"
        elif lower.endswith(".webm"):
            content_type = "video/webm"
        elif lower.endswith(".gif"):
            content_type = "image/gif"
        elif lower.endswith(".webp"):
            content_type = "image/webp"
        file_size = os.path.getsize(video_path)
        range_header = self.headers.get("Range", "").strip()
        if not range_header.startswith("bytes="):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(file_size))
            self.end_headers()
            with open(video_path, "rb") as handle:
                try:
                    self.wfile.write(handle.read())
                except (BrokenPipeError, ConnectionResetError):
                    return
            return

        range_spec = range_header[6:].split(",", 1)[0].strip()
        start_text, _, end_text = range_spec.partition("-")
        if not _:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else (file_size - 1)
            else:
                suffix_len = int(end_text)
                if suffix_len <= 0:
                    raise ValueError()
                start = max(0, file_size - suffix_len)
                end = file_size - 1
        except ValueError:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        if start < 0 or end < start or start >= file_size:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        end = min(end, file_size - 1)
        length = (end - start) + 1

        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(video_path, "rb") as handle:
            handle.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _stream_mjpeg(self):
        """Stream the latest annotated frame as multipart MJPEG for the browser <img> tag."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            width = config.width or 640
            height = config.height or 360
            svg = make_placeholder_svg(width, height, "CAMERA FEED")
            body = svg.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if cv2 is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "OpenCV not available")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                payload_ref = config.latest_jpeg
                payload = None if payload_ref is None else bytes(payload_ref)
                if payload is None:
                    frame_ready_event = getattr(config, "frame_ready_event", None)
                    if frame_ready_event is not None:
                        frame_ready_event.wait(timeout=0.05)
                    else:
                        time.sleep(0.05)
                    continue
                with config.settings_lock:
                    stream_fps = float(config.stream_fps)
                delay = 1.0 / max(1.0, stream_fps)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_preview_mjpeg(self, camera_index_value: str):
        """Stream one auxiliary preview camera as multipart MJPEG."""
        camera_manager = self._camera_manager()
        if camera_manager is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Multi-camera manager unavailable")
            return
        try:
            camera_index = int(camera_index_value)
        except (TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid camera index")
            return
        preview = camera_manager.get_camera(camera_index)
        if preview is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Camera not active")
            return
        config = self.server.config
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                preview = camera_manager.get_camera(camera_index)
                if preview is None:
                    return
                payload = preview.get_jpeg()
                if payload is None:
                    preview.frame_ready_event.wait(timeout=0.05)
                    continue
                with config.settings_lock:
                    stream_fps = float(config.stream_fps)
                delay = 1.0 / max(1.0, stream_fps)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_snapshot(self, camera_index_value=None):
        """Return the current camera frame as a single JPEG image."""
        try:
            source = self._resolve_camera_source(camera_index_value)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except LookupError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        payload = source["jpeg"]
        if payload is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available")
            return
        body = bytes(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_classes(self):
        """Return the merged list of known food/drink class names."""
        config = self.server.config
        names = set(FOOD_CLASS_NAMES)
        try:
            trained = read_class_map(config.class_map_path)
            names.update(trained.keys())
        except Exception:
            pass
        self._send_json(sorted(names), HTTPStatus.OK)

    def _create_manual_annotation(self):
        """Save a user-drawn bounding box as a new alert for accept/reject training."""
        config = self.server.config
        if cv2 is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "OpenCV not available")
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        bbox = payload.get("bbox")
        class_name = str(payload.get("class_name", "")).strip().lower()
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            self.send_error(HTTPStatus.BAD_REQUEST, "bbox must be [x1, y1, x2, y2] ratios")
            return
        if not class_name or len(class_name) > 64 or re.search(r'[/\\]', class_name):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid class_name")
            return
        try:
            ratios = [float(v) for v in bbox]
        except (TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "bbox values must be numbers")
            return
        for r in ratios:
            if r < 0.0 or r > 1.0:
                self.send_error(HTTPStatus.BAD_REQUEST, "bbox ratios must be 0.0-1.0")
                return

        try:
            source = self._resolve_camera_source(payload.get("camera_index"))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except LookupError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return

        frame = source["frame"]
        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available")
            return
        frame = frame.copy()
        h, w = frame.shape[:2]

        x1 = ratios[0] * w
        y1 = ratios[1] * h
        x2 = ratios[2] * w
        y2 = ratios[3] * h
        left, top, right, bottom = clamp_box([x1, y1, x2, y2], w, h)

        # Add margin around the crop for context (20% on each side).
        box_w = right - left
        box_h = bottom - top
        margin_x = int(box_w * 0.2)
        margin_y = int(box_h * 0.2)
        crop_left = max(0, left - margin_x)
        crop_top = max(0, top - margin_y)
        crop_right = min(w, right + margin_x)
        crop_bottom = min(h, bottom + margin_y)
        crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        if crop.size == 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "Drawn box is too small")
            return

        # Draw the bounding box on the crop.
        rel_left = left - crop_left
        rel_top = top - crop_top
        rel_right = right - crop_left
        rel_bottom = bottom - crop_top
        cv2.rectangle(crop, (rel_left, rel_top), (rel_right, rel_bottom), (0, 180, 255), 2)
        label = class_name
        cv2.putText(crop, label, (rel_left, max(rel_top - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        # Compute normalised bbox within the crop.
        crop_h, crop_w = crop.shape[:2]
        cx = (rel_left + rel_right) / 2.0 / crop_w
        cy = (rel_top + rel_bottom) / 2.0 / crop_h
        bw = (rel_right - rel_left) / crop_w
        bh = (rel_bottom - rel_top) / crop_h

        alert_id = uuid4().hex[:12]
        safe_cls = re.sub(r'[^a-z0-9_]', '_', class_name)
        snippet_name = f"manual_{alert_id}_0_{safe_cls}.jpg"
        snippet_dir = config.snippet_dir
        if snippet_dir:
            os.makedirs(snippet_dir, exist_ok=True)
            snippet_path = os.path.join(snippet_dir, snippet_name)
            cv2.imwrite(snippet_path, crop)

        detection = {
            "class_name": class_name,
            "confidence": 1.0,
            "bbox_xyxy": [left, top, right, bottom],
            "snippet_file": snippet_name,
            "snippet_bbox_xywhn": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
            "training_exported": False,
        }

        zone = str(source.get("zone", getattr(config, "camera_zone", "Zone A")))
        camera_index = int(source.get("camera_index", getattr(config, "camera_index", 0)))
        alert = {
            "id": alert_id,
            "status": "new",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "alert_reason": "manual_annotation",
            "zone": zone,
            "camera_index": camera_index,
            "frame_size": {"width": w, "height": h},
            "consumption_motion_detected": False,
            "consumption_motion_score": 0.0,
            "hand_to_mouth_source": "none",
            "hand_to_mouth_event_count": 0,
            "video_file": None,
            "video_mime": "",
            "detections": [detection],
        }

        with config.alert_lock:
            append_alert(
                config.alert_log,
                alert,
                summary_csv_path=config.detection_summary_csv,
            )
        self._send_json({"ok": True, "alert_id": alert_id}, HTTPStatus.CREATED)

    def log_message(self, format, *args):
        return

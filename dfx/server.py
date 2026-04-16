"""HTTP request handler serving the dashboard page, JSON APIs, and MJPEG stream."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from dfx.alerts import (
    build_consumption_stats,
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
    start_training_job,
    training_status_snapshot,
)

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
        if parsed.path.startswith("/snippets/"):
            self._send_snippet(parsed.path.removeprefix("/snippets/"))
            return
        if parsed.path.startswith("/videos/"):
            self._send_video(parsed.path.removeprefix("/videos/"))
            return
        if parsed.path == "/stream":
            self._stream_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
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

    def _update_settings(self):
        """Apply settings posted from the dashboard form."""
        config: DashboardConfig = self.server.config
        try:
            payload = self._read_json_body()
            updated = update_runtime_settings(config, payload)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _reset_settings(self):
        """Restore runtime settings to the startup defaults."""
        config: DashboardConfig = self.server.config
        updated = reset_runtime_settings(config)
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
                with config.frame_lock:
                    payload = None if config.latest_jpeg is None else bytes(config.latest_jpeg)
                if payload is None:
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

    def log_message(self, format, *args):
        return

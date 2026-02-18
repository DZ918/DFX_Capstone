import argparse
import json
import os
import random
import threading
import time
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Lab Food/Drink Monitor</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; margin: 0; background: #f6f7fb; }
      header { padding: 16px 20px; background: #111827; color: #fff; }
      main { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; }
      .card { background: #fff; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); padding: 12px; }
      img { width: 100%; border-radius: 8px; background: #000; }
      h2 { margin: 8px 0 12px; font-size: 16px; }
      ul { list-style: none; padding: 0; margin: 0; }
      li { padding: 8px 0; border-bottom: 1px solid #eee; font-size: 14px; }
      .meta { color: #6b7280; font-size: 12px; }
      .badge { display: inline-block; padding: 2px 6px; border-radius: 999px; background: #fee2e2; color: #991b1b; font-size: 12px; margin-right: 6px; }
      @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <header>
      <strong>Lab Food/Drink Monitor</strong>
    </header>
    <main>
      <section class=\"card\">
        <h2>Live Camera Feed</h2>
        <img src=\"/stream\" alt=\"Live camera feed\" />
      </section>
      <section class=\"card\">
        <h2>Recent Alerts</h2>
        <ul id=\"alerts\"></ul>
      </section>
    </main>
    <script>
      async function refreshAlerts() {
        try {
          const res = await fetch('/alerts?limit=20');
          const data = await res.json();
          const list = document.getElementById('alerts');
          list.innerHTML = '';
          data.forEach(alert => {
            const li = document.createElement('li');
            const count = alert.detections ? alert.detections.length : 0;
            li.innerHTML = `<span class=\"badge\">${count} item(s)</span> ${alert.timestamp}` +
              `<div class=\"meta\">` +
              (alert.detections || []).map(d => `${d.class_name} (${d.confidence})`).join(', ') +
              `</div>`;
            list.appendChild(li);
          });
        } catch (err) {
          // Ignore transient errors.
        }
      }
      refreshAlerts();
      setInterval(refreshAlerts, 2000);
    </script>
  </body>
</html>
"""


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


def draw_detections(frame, detections):
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
    return annotated


def make_placeholder_svg(width: int, height: int, label: str) -> str:
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
        f"viewBox=\"0 0 {width} {height}\">\n"
        "  <rect width=\"100%\" height=\"100%\" fill=\"#121824\" />\n"
        "  <text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dominant-baseline=\"middle\" "
        "font-family=\"Arial, sans-serif\" font-size=\"32\" fill=\"#f3f4f6\">"
        f"{safe_label}</text>\n"
        "</svg>\n"
    )


def make_random_alerts(limit: int, frame_width: int, frame_height: int) -> list[dict]:
    alerts: list[dict] = []
    class_names = sorted(FOOD_CLASS_NAMES)
    for idx in range(max(1, limit)):
        det_count = random.randint(1, 3)
        detections = []
        for _ in range(det_count):
            class_name = random.choice(class_names)
            x1 = random.randint(0, max(0, frame_width - 60))
            y1 = random.randint(0, max(0, frame_height - 60))
            x2 = random.randint(x1 + 20, min(frame_width, x1 + 200))
            y2 = random.randint(y1 + 20, min(frame_height, y1 + 200))
            detections.append(
                {
                    "class_id": -1,
                    "class_name": class_name,
                    "confidence": round(random.uniform(0.5, 0.99), 2),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                }
            )
        alert_time = datetime.now() - timedelta(seconds=idx * 3)
        alerts.append(
            {
                "timestamp": alert_time.isoformat(timespec="seconds"),
                "frame_size": {"width": frame_width, "height": frame_height},
                "detections": detections,
            }
        )
    return alerts


class DashboardConfig:
    def __init__(
        self,
        model,
        alert_log,
        width,
        height,
        stream_fps,
        conf,
        iou,
        persist_frames,
        cooldown,
        clear_frames,
        test_mode,
    ):
        self.model = model
        self.alert_log = alert_log
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.conf = conf
        self.iou = iou
        self.persist_frames = persist_frames
        self.cooldown = cooldown
        self.clear_frames = clear_frames
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.stop = False
        self.consecutive = 0
        self.clear_count = 0
        self.armed = True
        self.last_alert_ts = 0.0
        self.test_mode = test_mode


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FoodDrinkDashboard/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/alerts":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", [50])[0])
            self._send_alerts(limit)
            return
        if parsed.path == "/stream":
            self._stream_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_alerts(self, limit):
        config: DashboardConfig = self.server.config
        if config.test_mode:
            frame_width = config.width or 640
            frame_height = config.height or 360
            alerts = make_random_alerts(limit, frame_width, frame_height)
            body = json.dumps(alerts, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        alerts = []
        if config.alert_log and os.path.exists(config.alert_log):
            try:
                with open(config.alert_log, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        alerts = data
            except (json.JSONDecodeError, OSError):
                alerts = []
        if limit > 0:
            alerts = alerts[-limit:]
        body = json.dumps(alerts, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self):
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
        delay = 1.0 / max(1.0, config.stream_fps)
        try:
            while True:
                with config.frame_lock:
                    frame = None if config.latest_frame is None else config.latest_frame.copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                payload = encoded.tobytes()
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


def camera_worker(config: DashboardConfig, cam_index: int):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for live camera mode.")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}.")

    allowed_ids = get_allowed_class_ids(config.model, FOOD_CLASS_NAMES)

    try:
        while not config.stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            try:
                results = config.model.predict(
                    frame,
                    verbose=False,
                    conf=config.conf,
                    iou=config.iou,
                    classes=allowed_ids if allowed_ids else None,
                )
            except TypeError:
                results = config.model.predict(
                    frame,
                    verbose=False,
                    conf=config.conf,
                    iou=config.iou,
                )
            result = results[0]
            detections = detections_from_result(result, allowed_names=FOOD_CLASS_NAMES)

            if detections:
                config.consecutive += 1
                config.clear_count = 0
            else:
                config.consecutive = 0
                config.clear_count += 1
                if config.clear_count >= max(1, config.clear_frames):
                    config.armed = True

            now = time.time()
            if (
                detections
                and config.consecutive >= max(1, config.persist_frames)
                and config.armed
                and (now - config.last_alert_ts) >= max(0.0, config.cooldown)
            ):
                alert = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                    "detections": detections,
                }
                append_alert(config.alert_log, alert)
                config.last_alert_ts = now
                config.armed = False

            annotated = draw_detections(frame, detections)
            if config.width or config.height:
                annotated = cv2.resize(
                    annotated,
                    (
                        config.width or annotated.shape[1],
                        config.height or annotated.shape[0],
                    ),
                )

            with config.frame_lock:
                config.latest_frame = annotated
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Camera dashboard with live alerts")
    parser.add_argument("--test", action="store_true", help="Run with synthetic feed/alerts")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--alert-log", default="alerts.json", help="Alert JSON path")
    parser.add_argument("--width", type=int, default=0, help="Resize width (0 = original)")
    parser.add_argument("--height", type=int, default=0, help="Resize height (0 = original)")
    parser.add_argument("--fps", type=int, default=10, help="Stream FPS")
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
    args = parser.parse_args()

    if not args.test:
        if YOLO is None:
            raise RuntimeError("ultralytics is required unless --test is used.")
        if cv2 is None:
            raise RuntimeError("opencv-python is required unless --test is used.")

    model = None if args.test else YOLO(args.model)

    config = DashboardConfig(
        model=model,
        alert_log=args.alert_log,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        conf=args.conf,
        iou=args.iou,
        persist_frames=args.persist_frames,
        cooldown=args.cooldown,
        clear_frames=args.clear_frames,
        test_mode=args.test,
    )

    if not args.test:
        worker = threading.Thread(target=camera_worker, args=(config, args.cam), daemon=True)
        worker.start()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.config = config
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        config.stop = True


if __name__ == "__main__":
    main()

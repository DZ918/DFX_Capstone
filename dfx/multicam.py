"""Manage auxiliary camera preview workers for the browser dashboard."""

from __future__ import annotations

import threading
import time

try:
    import cv2
except Exception:
    cv2 = None

from dfx.camera import _open_camera_capture
from dfx.detection import make_status_frame


class CameraPreview:
    """Capture one auxiliary camera stream on a background thread."""

    def __init__(
        self,
        camera_index: int,
        *,
        stream_fps: float,
        width: int,
        height: int,
        jpeg_quality: int,
    ):
        self.camera_index = int(camera_index)
        self.available = False
        self.error = "Connecting..."
        self.updated_at = 0.0
        self._latest_frame = None
        self._latest_jpeg = b""
        self._lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self.frame_ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._stream_fps = max(1.0, float(stream_fps))
        self._width = max(0, int(width))
        self._height = max(0, int(height))
        self._jpeg_quality = max(40, min(95, int(jpeg_quality)))
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"camera-preview-{self.camera_index}",
        )
        self._thread.start()

    def stop(self):
        """Stop the background capture loop for this preview camera."""
        self._stop_event.set()
        self.frame_ready_event.set()
        self._thread.join(timeout=5.0)

    def update_settings(
        self,
        *,
        stream_fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        jpeg_quality: int | None = None,
    ):
        """Apply updated stream settings without restarting the worker thread."""
        with self._settings_lock:
            if stream_fps is not None:
                self._stream_fps = max(1.0, float(stream_fps))
            if width is not None:
                self._width = max(0, int(width))
            if height is not None:
                self._height = max(0, int(height))
            if jpeg_quality is not None:
                self._jpeg_quality = max(40, min(95, int(jpeg_quality)))

    def get_jpeg(self) -> bytes | None:
        """Return the latest encoded frame snapshot."""
        with self._lock:
            payload = self._latest_jpeg
        return bytes(payload) if payload else None

    def get_frame(self):
        """Return the latest raw frame snapshot for annotation operations."""
        with self._lock:
            frame = self._latest_frame
        return None if frame is None else frame.copy()

    def snapshot(self) -> dict:
        """Return the UI-facing status for this preview worker."""
        with self._lock:
            return {
                "index": int(self.camera_index),
                "available": bool(self.available),
                "error": str(self.error or ""),
                "updated_at": float(self.updated_at),
            }

    def _set_state(self, payload: bytes | None, *, available: bool, error: str, frame=None):
        with self._lock:
            self._latest_frame = None if frame is None else frame.copy()
            self._latest_jpeg = payload or b""
            self.available = bool(available)
            self.error = str(error or "")
            self.updated_at = time.time()
        self.frame_ready_event.set()

    def _status_dimensions(self) -> tuple[int, int]:
        with self._settings_lock:
            width = max(320, int(self._width or 640))
            height = max(180, int(self._height or 360))
        return width, height

    def _publish_status(self, label: str):
        width, height = self._status_dimensions()
        frame = make_status_frame(width, height, label)
        if frame is None or cv2 is None:
            self._set_state(None, available=False, error=label)
            return
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        self._set_state(encoded.tobytes() if ok else None, available=False, error=label, frame=None)

    def _encode_frame(self, frame) -> bytes | None:
        if cv2 is None:
            return None
        with self._settings_lock:
            jpeg_quality = int(self._jpeg_quality)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def _run(self):
        reconnect_delay = 1.0
        self._publish_status("Connecting...")
        while not self._stop_event.is_set():
            if cv2 is None:
                self._publish_status("OpenCV unavailable")
                self._stop_event.wait(timeout=reconnect_delay)
                continue

            cap = _open_camera_capture(self.camera_index)
            if cap is None:
                self._publish_status(f"Camera {self.camera_index} unavailable")
                self._stop_event.wait(timeout=reconnect_delay)
                continue
            if not cap.isOpened():
                cap.release()
                self._publish_status(f"Camera {self.camera_index} unavailable")
                self._stop_event.wait(timeout=reconnect_delay)
                continue

            self._set_state(self.get_jpeg(), available=True, error="", frame=None)

            while not self._stop_event.is_set():
                loop_started_at = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    self._publish_status(f"Camera {self.camera_index} reconnecting")
                    break
                payload = self._encode_frame(frame)
                if payload is not None:
                    self._set_state(payload, available=True, error="", frame=frame)
                with self._settings_lock:
                    stream_fps = float(self._stream_fps)
                delay = 1.0 / max(1.0, stream_fps)
                remaining = delay - (time.perf_counter() - loop_started_at)
                if remaining > 0:
                    self._stop_event.wait(timeout=remaining)

            cap.release()
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=reconnect_delay)


class CameraManager:
    """Track auxiliary preview cameras alongside the primary dashboard feed."""

    def __init__(
        self,
        primary_camera_index: int,
        *,
        stream_fps: float,
        width: int,
        height: int,
        jpeg_quality: int,
    ):
        self._lock = threading.Lock()
        self._preview_cameras: dict[int, CameraPreview] = {}
        self._primary_camera_index = int(primary_camera_index)
        self._stream_fps = float(stream_fps)
        self._width = int(width)
        self._height = int(height)
        self._jpeg_quality = int(jpeg_quality)

    def update_primary_camera(self, camera_index: int):
        """Move the primary role to a new camera and release duplicate previews."""
        preview = None
        with self._lock:
            self._primary_camera_index = int(camera_index)
            preview = self._preview_cameras.pop(self._primary_camera_index, None)
        if preview is not None:
            preview.stop()

    def update_stream_settings(self, *, stream_fps: float, width: int, height: int, jpeg_quality: int):
        """Propagate stream-related runtime settings to active preview workers."""
        with self._lock:
            self._stream_fps = float(stream_fps)
            self._width = int(width)
            self._height = int(height)
            self._jpeg_quality = int(jpeg_quality)
            previews = list(self._preview_cameras.values())
        for preview in previews:
            preview.update_settings(
                stream_fps=stream_fps,
                width=width,
                height=height,
                jpeg_quality=jpeg_quality,
            )

    def add_camera(self, camera_index: int) -> dict:
        """Start preview streaming for one additional camera index."""
        try:
            index = int(camera_index)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Camera index must be an integer"}
        if index < 0 or index > 99:
            return {"ok": False, "error": "Camera index must be between 0 and 99"}
        with self._lock:
            if index == self._primary_camera_index:
                return {"ok": False, "error": f"Camera {index} is already the primary feed"}
            if index in self._preview_cameras:
                return {"ok": False, "error": f"Camera {index} is already active"}
            preview = CameraPreview(
                index,
                stream_fps=self._stream_fps,
                width=self._width,
                height=self._height,
                jpeg_quality=self._jpeg_quality,
            )
            self._preview_cameras[index] = preview
        return {"ok": True, "index": index}

    def remove_camera(self, camera_index: int) -> dict:
        """Stop preview streaming for one auxiliary camera index."""
        try:
            index = int(camera_index)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Camera index must be an integer"}
        with self._lock:
            preview = self._preview_cameras.pop(index, None)
        if preview is None:
            return {"ok": False, "error": f"Camera {index} is not active"}
        preview.stop()
        return {"ok": True, "index": index}

    def get_camera(self, camera_index: int) -> CameraPreview | None:
        """Return one active preview worker by index."""
        with self._lock:
            preview = self._preview_cameras.get(int(camera_index))
        return preview

    def status_snapshot(self) -> list[dict]:
        """Return status snapshots for all active auxiliary preview cameras."""
        with self._lock:
            items = sorted(self._preview_cameras.items())
        return [preview.snapshot() for _, preview in items]

    def shutdown(self):
        """Stop and forget all active preview camera workers."""
        with self._lock:
            previews = list(self._preview_cameras.values())
            self._preview_cameras.clear()
        for preview in previews:
            preview.stop()
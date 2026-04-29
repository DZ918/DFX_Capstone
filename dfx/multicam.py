"""Manage auxiliary camera preview workers for the browser dashboard."""

from __future__ import annotations

import logging
import threading
import time

try:
    import cv2
except Exception:
    cv2 = None

from dfx.camera import _open_camera_capture
from dfx.detection import (
    detections_from_result,
    draw_detections,
    get_allowed_class_ids,
    make_status_frame,
)
from dfx.gpu import predict_with_fallback

logger = logging.getLogger(__name__)


class CameraPreview:
    """Capture one auxiliary camera stream on a background thread."""

    def __init__(
        self,
        camera_index: int,
        *,
        manager,
        stream_fps: float,
        width: int,
        height: int,
        jpeg_quality: int,
    ):
        self._manager = manager
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
        last_detections: list[dict] = []
        next_inference_at = 0.0
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
                # Cameras are physically mounted upside down; rotate before preview inference/streaming.
                frame = cv2.flip(frame, -1)
                detections = last_detections
                settings = self._manager.detection_settings_snapshot()
                if settings["enabled"]:
                    perf_now = time.perf_counter()
                    inference_due = (
                        settings["max_inference_fps"] <= 0.0 or perf_now >= next_inference_at
                    )
                    if inference_due:
                        if settings["max_inference_fps"] > 0.0:
                            next_inference_at = perf_now + (1.0 / max(0.1, settings["max_inference_fps"]))
                        else:
                            next_inference_at = 0.0
                        maybe_detections = self._manager.predict_detections(frame)
                        if maybe_detections is not None:
                            last_detections = maybe_detections
                            detections = maybe_detections
                    else:
                        detections = last_detections
                else:
                    last_detections = []
                    detections = []
                    next_inference_at = 0.0

                annotated_frame = draw_detections(frame, detections) if detections else frame
                payload = self._encode_frame(annotated_frame)
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
        model,
        model_lock,
        detection_enabled: bool,
        conf: float,
        iou: float,
        inference_imgsz: int,
        max_inference_fps: float,
        inference_device: str,
        tracked_class_names: set[str],
        allowed_class_names: set[str],
    ):
        self._lock = threading.Lock()
        self._preview_cameras: dict[int, CameraPreview] = {}
        self._primary_camera_index = int(primary_camera_index)
        self._stream_fps = float(stream_fps)
        self._width = int(width)
        self._height = int(height)
        self._jpeg_quality = int(jpeg_quality)
        self._model = model
        self._model_lock = model_lock
        self._detection_enabled = bool(detection_enabled)
        self._conf = float(conf)
        self._iou = float(iou)
        self._inference_imgsz = int(inference_imgsz)
        self._max_inference_fps = float(max_inference_fps)
        self._inference_device = str(inference_device or "cpu")
        self._tracked_class_names = {
            str(name).strip().lower() for name in tracked_class_names if str(name).strip()
        }
        self._allowed_class_names = {
            str(name).strip().lower() for name in allowed_class_names if str(name).strip()
        }
        self._allowed_ids = None
        self._last_inference_error = ""

    def update_primary_camera(self, camera_index: int):
        """Move the primary role to a new camera and release duplicate previews."""
        preview = None
        with self._lock:
            self._primary_camera_index = int(camera_index)
            preview = self._preview_cameras.pop(self._primary_camera_index, None)
        if preview is not None:
            preview.stop()

    def update_stream_settings(
        self,
        *,
        stream_fps: float,
        width: int,
        height: int,
        jpeg_quality: int,
        detection_enabled: bool | None = None,
        conf: float | None = None,
        iou: float | None = None,
        inference_imgsz: int | None = None,
        max_inference_fps: float | None = None,
        inference_device: str | None = None,
    ):
        """Propagate stream and inference settings to active preview workers."""
        with self._lock:
            self._stream_fps = float(stream_fps)
            self._width = int(width)
            self._height = int(height)
            self._jpeg_quality = int(jpeg_quality)
            if detection_enabled is not None:
                self._detection_enabled = bool(detection_enabled)
            if conf is not None:
                self._conf = float(conf)
            if iou is not None:
                self._iou = float(iou)
            if inference_imgsz is not None:
                self._inference_imgsz = int(inference_imgsz)
            if max_inference_fps is not None:
                self._max_inference_fps = float(max_inference_fps)
            if inference_device is not None:
                self._inference_device = str(inference_device or "cpu")
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
                manager=self,
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

    def update_runtime_detection_config(
        self,
        *,
        model=None,
        inference_device: str | None = None,
        tracked_class_names: set[str] | None = None,
        allowed_class_names: set[str] | None = None,
    ):
        """Update the shared live model or class filters after training finishes."""
        with self._lock:
            if model is not None and model is not self._model:
                self._model = model
                self._allowed_ids = None
            if inference_device is not None:
                self._inference_device = str(inference_device or "cpu")
            if tracked_class_names is not None:
                self._tracked_class_names = {
                    str(name).strip().lower() for name in tracked_class_names if str(name).strip()
                }
            if allowed_class_names is not None:
                normalized_allowed = {
                    str(name).strip().lower() for name in allowed_class_names if str(name).strip()
                }
                if normalized_allowed != self._allowed_class_names:
                    self._allowed_class_names = normalized_allowed
                    self._allowed_ids = None

    def detection_settings_snapshot(self) -> dict:
        """Return the current preview detection settings."""
        with self._lock:
            return {
                "enabled": bool(self._detection_enabled and self._model is not None),
                "max_inference_fps": float(self._max_inference_fps),
            }

    def predict_detections(self, frame) -> list[dict] | None:
        """Run one throttled prediction for a preview frame, or skip when the model is busy."""
        with self._lock:
            model = self._model
            model_lock = self._model_lock
            detection_enabled = bool(self._detection_enabled)
            conf = float(self._conf)
            iou = float(self._iou)
            inference_imgsz = int(self._inference_imgsz)
            inference_device = str(self._inference_device or "cpu")
            allowed_ids = self._allowed_ids
            allowed_class_names = set(self._allowed_class_names)
            tracked_class_names = set(self._tracked_class_names)

        if not detection_enabled or model is None or model_lock is None:
            return []
        if not model_lock.acquire(blocking=False):
            return None

        try:
            if allowed_ids is None:
                allowed_ids = get_allowed_class_ids(model, allowed_class_names)
                with self._lock:
                    self._allowed_ids = allowed_ids
            predict_kwargs = {
                "verbose": False,
                "conf": conf,
                "iou": iou,
                "imgsz": inference_imgsz,
                "classes": allowed_ids if allowed_ids else None,
                "device": inference_device,
            }
            results = predict_with_fallback(model, frame, **predict_kwargs)
            selected_device = str(
                getattr(model, "_dfx_inference_device_override", inference_device)
            ).strip() or "cpu"
            with self._lock:
                self._inference_device = selected_device
                self._last_inference_error = ""
        except Exception as exc:
            logger.warning("Preview inference failed for auxiliary camera: %s", exc)
            with self._lock:
                self._last_inference_error = str(exc)
            return []
        finally:
            model_lock.release()

        result = results[0]
        return detections_from_result(result, allowed_names=tracked_class_names)

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
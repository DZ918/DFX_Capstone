"""GPU detection, device selection, and optional TensorRT export for Jetson."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import platform
import site
import sysconfig
from typing import Any

logger = logging.getLogger(__name__)

_JETSON_RELEASE_PATH = Path("/etc/nv_tegra_release")
_JETSON_MODEL_PATH = Path("/proc/device-tree/model")
_JETSON_SYSTEM_LIBRARY_DIRS = (
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/aarch64-linux/lib",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu/tegra",
    "/usr/lib/aarch64-linux-gnu/nvidia",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except OSError:
        return ""


def is_jetson_linux() -> bool:
    """Return whether the current host looks like an NVIDIA Jetson device."""
    if platform.machine() != "aarch64" or platform.system() != "Linux":
        return False
    if "tegra" in platform.release().lower():
        return True
    if _JETSON_RELEASE_PATH.exists():
        return True
    model_text = _read_text(_JETSON_MODEL_PATH).lower()
    return "jetson" in model_text or "nvidia" in model_text


def _iter_python_library_roots() -> list[str]:
    paths = sysconfig.get_paths()
    candidates = [
        paths.get("purelib"),
        paths.get("platlib"),
        site.getusersitepackages(),
        str(Path.home() / ".local" / "lib"),
    ]
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    roots: list[str] = []
    for path in candidates:
        if not path or path in roots:
            continue
        roots.append(path)
    return roots


def get_jetson_gpu_library_paths() -> list[str]:
    """Discover Jetson CUDA/PyTorch library directories for the current Python."""
    if not is_jetson_linux():
        return []

    candidate_paths: list[str] = []
    override = os.environ.get("DFX_JETSON_GPU_LIB_PATHS", "")
    if override:
        candidate_paths.extend(path for path in override.split(":") if path)

    for root in _iter_python_library_roots():
        candidate_paths.extend(
            [
                os.path.join(root, "cusparselt", "lib"),
                os.path.join(root, "nvidia", "cusparselt", "lib"),
                os.path.join(root, "torch", "lib"),
            ]
        )

    candidate_paths.extend(_JETSON_SYSTEM_LIBRARY_DIRS)

    resolved: list[str] = []
    for path in candidate_paths:
        if not path:
            continue
        normalized = os.path.abspath(path)
        if normalized in resolved or not os.path.isdir(normalized):
            continue
        resolved.append(normalized)
    return resolved


def configure_jetson_gpu_env() -> list[str]:
    """Ensure Jetson CUDA-related library directories are visible to PyTorch."""
    if not is_jetson_linux():
        return []

    discovered = get_jetson_gpu_library_paths()
    current = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged: list[str] = []
    for path in [*discovered, *current]:
        normalized = os.path.abspath(path)
        if normalized in merged or not os.path.isdir(normalized):
            continue
        merged.append(normalized)
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    return merged


# Configure GPU-related library paths before torch is imported.
configure_jetson_gpu_env()


def get_best_device() -> str:
    """Return the best available compute device string for YOLO/PyTorch."""
    configure_jetson_gpu_env()
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = getattr(torch.version, "cuda", None)
            logger.info(
                "CUDA available: %s (torch %s, cuda %s)",
                device_name,
                torch.__version__,
                cuda_version,
            )
            return "cuda:0"
    except Exception as exc:
        logger.warning("CUDA detection failed: %s", exc)
    logger.info("Falling back to CPU inference")
    return "cpu"


def prepare_model_for_inference(model: Any, device: str | None = None) -> str:
    """Move a YOLO model to the selected device when the backend supports it."""
    selected_device = device or get_best_device()
    if model is None:
        return selected_device

    if selected_device.startswith("cuda"):
        try:
            import torch

            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    for candidate, label in ((model, "model"), (getattr(model, "model", None), "backend")):
        if candidate is None or not hasattr(candidate, "to"):
            continue
        try:
            candidate.to(selected_device)
            logger.info("Prepared %s for inference on %s", label, selected_device)
            break
        except Exception as exc:
            logger.warning("Could not move %s to %s: %s", label, selected_device, exc)
    return selected_device


def _build_predict_attempts(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a small set of compatible predict kwargs for older Ultralytics releases."""
    attempts: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for drop_keys in ((), ("classes",), ("device",), ("classes", "device")):
        attempt = {key: value for key, value in kwargs.items() if key not in drop_keys}
        signature = tuple(sorted(attempt))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        attempts.append(attempt)
    return attempts


def _predict_with_compatible_kwargs(model: Any, source: Any, kwargs: dict[str, Any]):
    """Run ``model.predict`` while tolerating older Ultralytics keyword support."""
    attempts = _build_predict_attempts(kwargs)

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return model.predict(source, **attempt), attempt
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return model.predict(source, **kwargs), kwargs


def _clear_cuda_state():
    """Release any cached CUDA allocations after an inference failure."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_cuda_oom_error(exc: Exception) -> bool:
    """Return whether an exception looks like a CUDA out-of-memory failure."""
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(exc).strip().lower()
    return "out of memory" in message or "cuda out of memory" in message


def predict_with_fallback(model: Any, source: Any, **kwargs):
    """Run ``model.predict`` with compatibility and CUDA OOM fallback handling."""
    attempt_kwargs = dict(kwargs)
    forced_device = str(getattr(model, "_dfx_inference_device_override", "")).strip()
    requested_device = str(attempt_kwargs.get("device", "")).strip()
    if forced_device and requested_device.startswith("cuda"):
        attempt_kwargs["device"] = forced_device

    try:
        results, used_kwargs = _predict_with_compatible_kwargs(model, source, attempt_kwargs)
        if "device" in used_kwargs:
            model._dfx_inference_device_override = str(used_kwargs["device"])
        return results
    except Exception as exc:
        active_device = str(attempt_kwargs.get("device", "")).strip().lower()
        if not active_device.startswith("cuda") or not _is_cuda_oom_error(exc):
            raise

        logger.warning("CUDA inference ran out of memory; retrying on CPU: %s", exc)
        _clear_cuda_state()
        if hasattr(model, "predictor"):
            try:
                model.predictor = None
            except Exception:
                pass
        try:
            prepare_model_for_inference(model, "cpu")
        except Exception as move_exc:
            logger.warning("Could not move model to CPU after CUDA OOM: %s", move_exc)

        cpu_kwargs = dict(kwargs)
        cpu_kwargs["device"] = "cpu"
        results, _ = _predict_with_compatible_kwargs(model, source, cpu_kwargs)
        model._dfx_inference_device_override = "cpu"
        return results


def export_tensorrt(model_path: str, imgsz: int = 640, half: bool = True) -> str | None:
    """Export a YOLO ``.pt`` model to TensorRT ``.engine`` format."""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        export_path = model.export(format="engine", imgsz=imgsz, half=half)
        logger.info("TensorRT export complete: %s", export_path)
        return str(export_path)
    except Exception as exc:
        logger.warning("TensorRT export failed: %s", exc)
        return None

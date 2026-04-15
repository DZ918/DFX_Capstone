"""GPU detection, device selection, and optional TensorRT export for Jetson."""

import logging
import os
import platform

logger = logging.getLogger(__name__)


def _ensure_jetson_gpu_libs() -> None:
    """Ensure NVIDIA GPU libraries are in LD_LIBRARY_PATH on Jetson.

    This is a workaround for when the activation script isn't sourced.
    """
    if platform.machine() != "aarch64" or platform.system() != "Linux":
        return
    if "tegra" not in platform.release():
        return

    # Jetson-specific GPU library paths
    gpu_lib_paths = [
        "/home/user/.local/lib/python3.10/site-packages/cusparselt/lib",
        "/home/user/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib",
        "/home/user/.local/lib",
    ]

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [p for p in gpu_lib_paths if p and os.path.isdir(p)]

    if new_paths:
        if current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + ":" + current_ld_path
        else:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths)


# Set up GPU library paths before importing torch
_ensure_jetson_gpu_libs()


def get_best_device() -> str:
    """Return the best available compute device string for YOLO/PyTorch.

    Returns ``'cuda:0'`` if CUDA is available, otherwise ``'cpu'``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA available: %s (torch %s)", device_name, torch.__version__)
            return "cuda:0"
    except Exception as exc:
        logger.warning("CUDA detection failed: %s", exc)
    logger.info("Falling back to CPU inference")
    return "cpu"


def export_tensorrt(model_path: str, imgsz: int = 640, half: bool = True) -> str | None:
    """Export a YOLO ``.pt`` model to TensorRT ``.engine`` format.

    Returns the path to the exported engine file, or ``None`` on failure.
    Only meaningful on NVIDIA platforms with TensorRT installed.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        export_path = model.export(format="engine", imgsz=imgsz, half=half)
        logger.info("TensorRT export complete: %s", export_path)
        return str(export_path)
    except Exception as exc:
        logger.warning("TensorRT export failed: %s", exc)
        return None

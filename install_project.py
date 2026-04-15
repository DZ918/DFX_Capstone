"""Bootstrap this project on a new device.

Creates a virtual environment, installs Python dependencies, prepares runtime
folders/files, and makes sure the default YOLO model is available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


MIN_PYTHON = (3, 9)


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    pretty = " ".join(cmd)
    where = f" (cwd={cwd})" if cwd else ""
    log(f"[install] {pretty}{where}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        log(f"[install] Creating virtual environment at {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
    else:
        log(f"[install] Reusing existing virtual environment at {venv_dir}")
    python_path = get_venv_python(venv_dir)
    if not python_path.exists():
        fail(f"Virtual environment python not found: {python_path}")
    return python_path


def install_requirements(python_path: Path, requirements_path: Path) -> None:
    run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools<82",
            "wheel",
        ]
    )
    run([str(python_path), "-m", "pip", "install", "-r", str(requirements_path)])


def install_cuda_pytorch(python_path: Path) -> None:
    """Install CUDA-enabled PyTorch on Jetson (aarch64 Linux) from NVIDIA's index.

    On non-Jetson platforms this is a no-op; the generic PyTorch from
    requirements.txt is used instead.
    """
    import platform

    if platform.machine() != "aarch64" or platform.system() != "Linux":
        log("[install] Not a Jetson/aarch64 platform, skipping CUDA PyTorch install")
        return
    # Check for the Jetson-specific kernel suffix.
    if "tegra" not in platform.release():
        log("[install] Kernel does not look like Jetson (no -tegra), skipping CUDA PyTorch")
        return
    nvidia_index = "https://developer.download.nvidia.com/compute/redist/jp/v60"
    log("[install] Installing CUDA-enabled PyTorch for Jetson from NVIDIA index")
    try:
        run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"--extra-index-url={nvidia_index}",
                "torch",
                "torchvision",
            ]
        )
    except subprocess.CalledProcessError:
        log(
            "[install] WARNING: CUDA PyTorch install failed. "
            "Inference will fall back to CPU. You can install manually from "
            "https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
        )


def rebuild_torchvision_for_jetson(python_path: Path) -> None:
    """Rebuild torchvision from source to ensure compatibility with Jetson torch.

    Pre-built wheels often have incompatibilities. Building from source ensures
    the version matches the installed torch correctly.
    """
    import platform
    import tempfile

    if platform.machine() != "aarch64" or platform.system() != "Linux":
        return
    if "tegra" not in platform.release():
        return

    log("[install] Rebuilding torchvision from source for Jetson compatibility")
    try:
        # Remove pre-built torchvision to avoid conflicts
        run([str(python_path), "-m", "pip", "uninstall", "torchvision", "-y"])

        # Download and build from source
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run(
                [
                    "git",
                    "clone",
                    "--branch",
                    "v0.20.0",
                    "--depth",
                    "1",
                    "https://github.com/pytorch/vision.git",
                    str(tmp_path / "vision"),
                ]
            )
            # Build torchvision with CUDA support
            env = os.environ.copy()
            env["FORCE_CUDA"] = "1"
            run(
                [str(python_path), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."],
                cwd=tmp_path / "vision",
            )
    except subprocess.CalledProcessError as exc:
        log(f"[install] WARNING: Building torchvision from source failed: {exc}")
        log("[install] Attempting to reinstall pre-built torchvision")
        try:
            run([str(python_path), "-m", "pip", "install", "torchvision"])
        except subprocess.CalledProcessError:
            log(
                "[install] ERROR: Could not install torchvision. "
                "You may need to build it manually from "
                "https://github.com/pytorch/vision"
            )


def ensure_jetson_gpu_env(project_root: Path) -> None:
    """Create wrapper scripts for Jetson GPU support.

    Creates helper scripts that ensure LD_LIBRARY_PATH is set before running the app.
    """
    import platform

    if platform.machine() != "aarch64" or platform.system() != "Linux":
        return
    if "tegra" not in platform.release():
        return

    gpu_lib_path = "/home/user/.local/lib/python3.10/site-packages/cusparselt/lib"
    system_lib_path = "/home/user/.local/lib"

    # Create wrapper script for main.py
    main_wrapper = project_root / "run_main.sh"
    main_wrapper.write_text(
        f"""#!/bin/bash
export LD_LIBRARY_PATH="{gpu_lib_path}:{system_lib_path}:${{LD_LIBRARY_PATH:-}}"
exec python3 main.py "$@"
""",
        encoding="utf-8",
    )
    main_wrapper.chmod(0o755)
    log(f"[install] Created GPU wrapper script: {main_wrapper}")

    # Create wrapper script for dashboard.py
    dashboard_wrapper = project_root / "run_dashboard.sh"
    dashboard_wrapper.write_text(
        f"""#!/bin/bash
export LD_LIBRARY_PATH="{gpu_lib_path}:{system_lib_path}:${{LD_LIBRARY_PATH:-}}"
exec python3 dashboard.py "$@"
""",
        encoding="utf-8",
    )
    dashboard_wrapper.chmod(0o755)
    log(f"[install] Created GPU wrapper script: {dashboard_wrapper}")

    # Update .bashrc to set LD_LIBRARY_PATH for interactive shells
    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.exists():
        bashrc_content = bashrc_path.read_text(encoding="utf-8")
        marker = "# NVIDIA cusparseLt for PyTorch on Jetson"
        if marker not in bashrc_content:
            bashrc_content += f"""
{marker}
export LD_LIBRARY_PATH="{gpu_lib_path}:{system_lib_path}:${{LD_LIBRARY_PATH:-}}"
"""
            try:
                bashrc_path.write_text(bashrc_content, encoding="utf-8")
                log(f"[install] Updated ~/.bashrc with LD_LIBRARY_PATH for Jetson GPU")
            except Exception as exc:
                log(f"[install] WARNING: Could not update ~/.bashrc: {exc}")
    for relative_dir in (
        "snippets",
        "training_data",
        "training_data/dataset/images",
        "training_data/dataset/labels",
        "training_data/runs",
    ):
        path = project_root / relative_dir
        path.mkdir(parents=True, exist_ok=True)
        log(f"[install] Ensured directory: {path}")

    alerts_path = project_root / "alerts.json"
    if not alerts_path.exists():
        alerts_path.write_text("[]\n", encoding="utf-8")
        log(f"[install] Created file: {alerts_path}")
    else:
        try:
            data = json.loads(alerts_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                fail(f"{alerts_path} exists but is not a JSON list.")
        except json.JSONDecodeError as exc:
            fail(f"{alerts_path} is not valid JSON: {exc}")


def maybe_download_model(project_root: Path, python_path: Path, model_arg: str) -> None:
    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if model_path.exists():
        log(f"[install] Model already present: {model_path}")
        return

    if model_path.suffix != ".pt" or not model_path.name.startswith("yolo"):
        log(
            "[install] Model file is missing and does not look like a built-in Ultralytics "
            f"weight: {model_path}"
        )
        log("[install] Copy your custom model to that path before running the app.")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[install] Downloading default model weights to {model_path}")
    run(
        [str(python_path), "-c", f"from ultralytics import YOLO; YOLO({model_path.name!r})"],
        cwd=model_path.parent,
    )


def activation_hint(venv_dir: Path) -> str:
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "activate")
    return f"source {venv_dir / 'bin' / 'activate'}"


def print_next_steps(venv_dir: Path) -> None:
    log("")
    log("[install] Setup complete.")
    log(f"[install] Activate the environment with: {activation_hint(venv_dir)}")
    log("[install] Run the dashboard with: python dashboard.py")
    log("[install] Run the webcam demo with: python main.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install this OpenCV + YOLO project on a new device")
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Virtual environment directory to create/use",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Default model path to verify/download",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Do not try to download the default model if it is missing",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        fail(
            "Python 3.9 or newer is required. "
            f"Detected {sys.version_info.major}.{sys.version_info.minor}."
        )

    args = parse_args()
    project_root = Path(__file__).resolve().parent
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        fail(f"Missing requirements file: {requirements_path}")

    venv_dir = Path(args.venv)
    if not venv_dir.is_absolute():
        venv_dir = project_root / venv_dir

    python_path = create_venv(venv_dir)
    install_requirements(python_path, requirements_path)
    install_cuda_pytorch(python_path)
    rebuild_torchvision_for_jetson(python_path)
    ensure_jetson_gpu_env(project_root)
    ensure_runtime_layout(project_root)
    if not args.skip_model_download:
        maybe_download_model(project_root, python_path, args.model)
    print_next_steps(venv_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

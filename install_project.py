"""Bootstrap this project on a new device.

Creates a virtual environment, installs Python dependencies, prepares runtime
folders/files, and makes sure the default YOLO model is available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


MIN_PYTHON = (3, 9)


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def log(message: str) -> None:
    print(message, flush=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except OSError:
        return ""


def is_jetson_linux() -> bool:
    if platform.machine() != "aarch64" or platform.system() != "Linux":
        return False
    if "tegra" in platform.release().lower():
        return True
    if Path("/etc/nv_tegra_release").exists():
        return True
    model_text = _read_text(Path("/proc/device-tree/model")).lower()
    return "jetson" in model_text or "nvidia" in model_text


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    pretty = " ".join(cmd)
    where = f" (cwd={cwd})" if cwd else ""
    log(f"[install] {pretty}{where}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        log(f"[install] Creating virtual environment at {venv_dir}")
        use_system_packages = is_jetson_linux()
        if use_system_packages:
            log(
                "[install] Jetson detected; enabling system-site-packages so the "
                "NVIDIA CUDA torch install remains visible inside the virtual environment"
            )
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=use_system_packages)
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


def python_can_use_cuda(python_path: Path) -> bool:
    result = subprocess.run(
        [
            str(python_path),
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def install_cuda_pytorch(python_path: Path, torch_wheel: str | None = None) -> None:
    """Install a Jetson-compatible CUDA PyTorch wheel when one is provided."""
    if not is_jetson_linux():
        log("[install] Not a Jetson/aarch64 platform, skipping CUDA PyTorch install")
        return
    if python_can_use_cuda(python_path):
        log("[install] CUDA-enabled torch is already visible inside the virtual environment")
        return

    wheel = (torch_wheel or os.environ.get("TORCH_INSTALL") or os.environ.get("DFX_TORCH_WHEEL") or "").strip()
    if not wheel:
        log(
            "[install] WARNING: CUDA torch is not available in this environment. "
            "If your Jetson already has NVIDIA torch installed system-wide, recreate the venv "
            "with this script so system-site-packages are visible. Otherwise provide the official "
            "NVIDIA wheel URL with --torch-wheel or TORCH_INSTALL."
        )
        log(
            "[install] See NVIDIA's Jetson PyTorch guide: "
            "https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html"
        )
        return

    log(f"[install] Installing Jetson CUDA PyTorch from {wheel}")
    try:
        run([str(python_path), "-m", "pip", "install", "--no-cache", wheel])
    except subprocess.CalledProcessError:
        log(
            "[install] WARNING: Jetson CUDA PyTorch wheel install failed. "
            "Verify that the wheel matches your JetPack and Python version."
        )
        return

    if python_can_use_cuda(python_path):
        log("[install] CUDA-enabled torch is now available")
    else:
        log(
            "[install] WARNING: torch installed but CUDA is still unavailable. "
            "Check that the wheel matches your JetPack release."
        )


def rebuild_torchvision_for_jetson(python_path: Path) -> None:
    """Rebuild torchvision from source to ensure compatibility with Jetson torch.

    Pre-built wheels often have incompatibilities. Building from source ensures
    the version matches the installed torch correctly.
    """
    import tempfile

    if not is_jetson_linux():
        return

    rebuild_requested = os.environ.get("DFX_REBUILD_TORCHVISION", "").strip().lower()
    if rebuild_requested not in {"1", "true", "yes", "on"}:
        log(
            "[install] Skipping torchvision source rebuild. Set DFX_REBUILD_TORCHVISION=1 "
            "and TORCHVISION_REF=<git tag> only if your Jetson image needs a custom build."
        )
        return

    vision_ref = os.environ.get("TORCHVISION_REF", "").strip()
    if not vision_ref:
        log(
            "[install] WARNING: DFX_REBUILD_TORCHVISION is set but TORCHVISION_REF is missing; "
            "skipping torchvision rebuild"
        )
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
                    vision_ref,
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
                env=env,
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


def _jetson_wrapper_script(target_script: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
if [[ -n \"${{PYTHON:-}}\" ]]; then
    PYTHON_BIN=\"${{PYTHON}}\"
elif [[ -x \"${{SCRIPT_DIR}}/.venv/bin/python\" ]]; then
    PYTHON_BIN=\"${{SCRIPT_DIR}}/.venv/bin/python\"
else
    PYTHON_BIN=\"python3\"
fi

GPU_LIB_PATHS=\"$($PYTHON_BIN - <<'PY'
import os
from pathlib import Path
import site
import sysconfig

candidates = []
override = os.environ.get("DFX_JETSON_GPU_LIB_PATHS", "")
if override:
    candidates.extend(path for path in override.split(":") if path)

paths = sysconfig.get_paths()
roots = [
    paths.get("purelib"),
    paths.get("platlib"),
    site.getusersitepackages(),
    str(Path.home() / ".local" / "lib"),
]
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass

for root in roots:
    if not root:
        continue
    candidates.extend(
        [
            os.path.join(root, "cusparselt", "lib"),
            os.path.join(root, "nvidia", "cusparselt", "lib"),
            os.path.join(root, "torch", "lib"),
        ]
    )

candidates.extend(
    [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
        "/usr/lib/aarch64-linux-gnu/nvidia",
    ]
)

existing = []
for path in candidates:
    if not path:
        continue
    normalized = os.path.abspath(path)
    if normalized in existing or not os.path.isdir(normalized):
        continue
    existing.append(normalized)

print(\":\".join(existing))
PY
)\"

if [[ -n \"$GPU_LIB_PATHS\" ]]; then
    export LD_LIBRARY_PATH=\"${{GPU_LIB_PATHS}}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}\"
fi

exec \"$PYTHON_BIN\" \"${{SCRIPT_DIR}}/{target_script}\" \"$@\"
"""


def ensure_jetson_gpu_env(project_root: Path) -> None:
    """Create wrapper scripts for Jetson GPU support."""
    if not is_jetson_linux():
        return

    main_wrapper = project_root / "run_main.sh"
    main_wrapper.write_text(_jetson_wrapper_script("main.py"), encoding="utf-8")
    main_wrapper.chmod(0o755)
    log(f"[install] Created GPU wrapper script: {main_wrapper}")

    dashboard_wrapper = project_root / "run_dashboard.sh"
    dashboard_wrapper.write_text(_jetson_wrapper_script("dashboard.py"), encoding="utf-8")
    dashboard_wrapper.chmod(0o755)
    log(f"[install] Created GPU wrapper script: {dashboard_wrapper}")


def ensure_runtime_layout(project_root: Path) -> None:
    """Create runtime directories and validate the alert log file."""
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
    if is_jetson_linux():
        log("[install] Run the dashboard with: ./run_dashboard.sh")
        log("[install] Run the webcam demo with: ./run_main.sh")
    else:
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
    parser.add_argument(
        "--torch-wheel",
        help="Official NVIDIA Jetson PyTorch wheel URL or local path for this JetPack/Python build",
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
    install_cuda_pytorch(python_path, args.torch_wheel)
    rebuild_torchvision_for_jetson(python_path)
    ensure_jetson_gpu_env(project_root)
    ensure_runtime_layout(project_root)
    if not args.skip_model_download:
        maybe_download_model(project_root, python_path, args.model)
    print_next_steps(venv_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

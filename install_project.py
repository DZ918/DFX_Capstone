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


def ensure_runtime_layout(project_root: Path) -> None:
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
    ensure_runtime_layout(project_root)
    if not args.skip_model_download:
        maybe_download_model(project_root, python_path, args.model)
    print_next_steps(venv_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

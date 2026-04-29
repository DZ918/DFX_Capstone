#!/usr/bin/env python3
"""Import a Roboflow soda-can dataset and fine-tune the active DFX model.

Workflow:
1) Read a Roboflow YOLOv8 dataset directory.
2) Rewrite every label class ID to the target class ID (default: 7 / soda_can).
3) Copy images + remapped labels into training_data/dataset/images and labels.
4) Fine-tune training_data/runs/accepted/weights/best.pt on the mixed dataset.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover - startup dependency guard
    raise RuntimeError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_FILE_CANDIDATES = ("classes.txt", "labels.txt", "obj.names")


def _log(message: str) -> None:
    print(f"[import-train] {message}", flush=True)


def _resolve_path(path_value: str, project_root: Path) -> Path:
    raw = Path(path_value).expanduser()
    return raw if raw.is_absolute() else (project_root / raw)


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return token.strip("_") or "sample"


def _is_jetson_linux() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    model_path = Path("/proc/device-tree/model")
    try:
        model_text = model_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "jetson" in model_text


def _load_yolo_class():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - startup dependency guard
        raise RuntimeError(
            "ultralytics is required for training. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def _is_cuda_allocator_issue(message: str) -> bool:
    text = str(message or "").lower()
    markers = (
        "nvml_success",
        "cudacachingallocator",
        "cudaallocatorconfig.cpp",
        "allocator backend parsed at runtime",
        "cublas_status_alloc_failed",
        "cuda error:",
        "cuda out of memory",
        "driver shutting down",
        "allocation on device",
        "outofmemoryerror",
    )
    return any(marker in text for marker in markers)


def _patch_pin_memory_for_jetson() -> object | None:
    """Disable Tensor.pin_memory on Jetson unified memory to reduce allocation pressure."""
    if not _is_jetson_linux():
        return None
    try:
        import torch
    except Exception as exc:
        _log(f"Warning: torch import failed while disabling pin_memory: {exc}")
        return None
    original = getattr(torch.Tensor, "pin_memory", None)
    if original is None:
        return None
    torch.Tensor.pin_memory = lambda self, *a, **kw: self  # type: ignore[assignment]
    _log("Jetson mode: disabled Tensor.pin_memory for dataloader safety")
    return original


def _restore_pin_memory(original: object | None) -> None:
    if original is None:
        return
    try:
        import torch

        torch.Tensor.pin_memory = original  # type: ignore[assignment]
    except Exception:
        pass


def _find_image_for_label(label_path: Path) -> Path | None:
    stem = label_path.stem
    for suffix in IMAGE_EXTENSIONS:
        candidate = label_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    parts = label_path.parts
    label_indices = [idx for idx, part in enumerate(parts) if part.lower() == "labels"]
    for index in label_indices:
        prefix = Path(*parts[:index])
        suffix_parts = parts[index + 1 :]
        for extension in IMAGE_EXTENSIONS:
            candidate = prefix / "images" / Path(*suffix_parts)
            candidate = candidate.with_suffix(extension)
            if candidate.exists():
                return candidate
    return None


def _collect_yolo_pairs(dataset_root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    pairs: list[tuple[Path, Path]] = []
    missing_image_labels: list[Path] = []
    for label_path in sorted(dataset_root.rglob("*.txt")):
        if not label_path.is_file():
            continue
        if label_path.name.lower() in CLASS_FILE_CANDIDATES:
            continue
        image_path = _find_image_for_label(label_path)
        if image_path is None:
            missing_image_labels.append(label_path)
            continue
        pairs.append((image_path, label_path))
    return pairs, missing_image_labels


def _read_class_names(data_yaml_path: Path) -> dict[int, str]:
    try:
        payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not read data.yaml at {data_yaml_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Invalid YAML in {data_yaml_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a YAML mapping in {data_yaml_path}")

    raw_names = payload.get("names")
    if isinstance(raw_names, list):
        return {idx: str(name).strip() for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            names[idx] = str(value).strip()
        return names
    raise RuntimeError(f"Could not parse 'names' from {data_yaml_path}")


def _rewrite_label_to_target_class(
    label_source: Path,
    label_dest: Path,
    target_class_id: int,
) -> tuple[Counter[int], int, int, list[str]]:
    """Rewrite one label file to target class ID and return stats.

    Returns:
    - source class histogram
    - mapped annotation count
    - invalid line count
    - preview lines
    """
    try:
        lines = label_source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Failed to read label file {label_source}: {exc}") from exc

    source_counts: Counter[int] = Counter()
    rewritten_lines: list[str] = []
    invalid_lines = 0
    preview_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            invalid_lines += 1
            continue
        try:
            source_class_id = int(float(parts[0]))
        except (TypeError, ValueError):
            invalid_lines += 1
            continue
        source_counts[source_class_id] += 1
        parts[0] = str(int(target_class_id))
        mapped_line = " ".join(parts)
        rewritten_lines.append(mapped_line)
        if len(preview_lines) < 2:
            preview_lines.append(f"{source_class_id} -> {target_class_id} | {mapped_line}")

    payload = "\n".join(rewritten_lines)
    if payload:
        payload += "\n"
    try:
        label_dest.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write label file {label_dest}: {exc}") from exc

    return source_counts, len(rewritten_lines), invalid_lines, preview_lines


def _merge_roboflow_dataset(
    roboflow_root: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
    target_class_id: int,
) -> tuple[list[Path], Counter[int], int, int, int, list[str]]:
    pairs, missing_image_labels = _collect_yolo_pairs(roboflow_root)
    if not pairs:
        raise RuntimeError(
            f"No valid YOLO label/image pairs found in {roboflow_root}. "
            "Expected labels with matching images."
        )

    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    import_token = _sanitize_token(f"rf_{roboflow_root.name}_{int(time.time())}")
    imported_label_paths: list[Path] = []
    source_class_histogram: Counter[int] = Counter()
    total_mapped_annotations = 0
    total_invalid_lines = 0
    empty_label_files = 0
    preview: list[str] = []

    _log(f"Found {len(pairs)} label/image pairs under {roboflow_root}")
    if missing_image_labels:
        _log(f"Warning: {len(missing_image_labels)} label files had no matching image and were skipped")

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        try:
            rel_stem = image_path.relative_to(roboflow_root).with_suffix("")
            source_token = _sanitize_token(str(rel_stem).replace("/", "_"))
        except ValueError:
            source_token = _sanitize_token(image_path.stem)

        stem = f"{import_token}_{index:06d}_{source_token}"
        image_dest = target_images_dir / f"{stem}{image_path.suffix.lower() or '.jpg'}"
        label_dest = target_labels_dir / f"{stem}.txt"

        try:
            shutil.copy2(image_path, image_dest)
        except OSError as exc:
            _log(f"Warning: failed to copy image {image_path}: {exc}")
            continue

        source_counts, mapped_count, invalid_count, preview_lines = _rewrite_label_to_target_class(
            label_source=label_path,
            label_dest=label_dest,
            target_class_id=int(target_class_id),
        )
        source_class_histogram.update(source_counts)
        total_mapped_annotations += int(mapped_count)
        total_invalid_lines += int(invalid_count)
        if mapped_count == 0:
            empty_label_files += 1
        imported_label_paths.append(label_dest)

        if len(preview) < 8 and preview_lines:
            for line in preview_lines:
                if len(preview) >= 8:
                    break
                preview.append(f"{label_path.name}: {line}")

        if index % 250 == 0:
            _log(f"Imported {index}/{len(pairs)} files...")

    if not imported_label_paths:
        raise RuntimeError("No files were imported. Aborting training.")

    return (
        imported_label_paths,
        source_class_histogram,
        total_mapped_annotations,
        total_invalid_lines,
        empty_label_files,
        preview,
    )


def _verify_imported_labels(imported_label_paths: list[Path], target_class_id: int) -> int:
    """Ensure all non-empty annotation lines now use the requested target class ID."""
    checked_annotations = 0
    violations: list[str] = []

    for label_path in imported_label_paths:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"Failed to verify {label_path}: {exc}") from exc
        for line_no, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except (TypeError, ValueError):
                violations.append(f"{label_path}:{line_no} non-numeric class id '{parts[0]}'")
                continue
            checked_annotations += 1
            if class_id != int(target_class_id):
                violations.append(
                    f"{label_path}:{line_no} expected class {target_class_id}, found {class_id}"
                )
            if len(violations) >= 10:
                break
        if len(violations) >= 10:
            break

    if violations:
        preview = "\n".join(violations)
        raise RuntimeError(
            "Class-ID verification failed after remap. Sample issues:\n"
            f"{preview}"
        )
    return checked_annotations


def _train_model(
    weights_path: Path,
    data_yaml_path: Path,
    runs_project_dir: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    lr0: float,
    lrf: float,
    freeze: int,
    patience: int,
    allow_cpu_fallback: bool,
) -> Path:
    if _is_jetson_linux():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    pin_memory_backup = _patch_pin_memory_for_jetson()

    try:
        YOLO = _load_yolo_class()
        model = YOLO(str(weights_path))

        train_kwargs = {
            "data": str(data_yaml_path),
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "project": str(runs_project_dir),
            "name": str(run_name),
            "exist_ok": True,
            "verbose": True,
            "device": str(device),
            "batch": int(batch),
            "workers": int(workers),
            "optimizer": "SGD",
            "lr0": float(lr0),
            "lrf": float(lrf),
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "freeze": int(max(0, freeze)),
            "patience": int(max(1, patience)),
            "amp": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 0,
            "val": True,
            "plots": False,
        }

        _log("Starting fine-tune with conservative anti-forgetting settings")
        _log(
            "Train args: "
            f"device={train_kwargs['device']}, batch={train_kwargs['batch']}, "
            f"workers={train_kwargs['workers']}, lr0={train_kwargs['lr0']}, "
            f"freeze={train_kwargs['freeze']}, epochs={train_kwargs['epochs']}"
        )

        try:
            result = model.train(**train_kwargs)
        except Exception as exc:
            if not allow_cpu_fallback or not _is_cuda_allocator_issue(str(exc)):
                raise
            _log("CUDA allocator issue detected; retrying training on CPU")
            retry_kwargs = dict(train_kwargs)
            retry_kwargs["device"] = "cpu"
            retry_kwargs["workers"] = 0
            retry_kwargs["batch"] = max(1, min(2, int(retry_kwargs["batch"])))
            retry_kwargs["amp"] = False
            result = model.train(**retry_kwargs)
    finally:
        _restore_pin_memory(pin_memory_backup)

    save_dir = Path(str(getattr(result, "save_dir", runs_project_dir / run_name)))
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"Training finished but best.pt was not found at {best_path}")
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a Roboflow soda-can dataset and fine-tune DFX best.pt"
    )
    parser.add_argument(
        "roboflow_dataset",
        help="Path to downloaded Roboflow YOLO dataset folder",
    )
    parser.add_argument(
        "--project-root",
        default="/home/user/DFX_Capstone",
        help="Project root used to resolve relative paths",
    )
    parser.add_argument(
        "--data-yaml",
        default="training_data/dataset/data.yaml",
        help="Existing training data.yaml used for fine-tuning",
    )
    parser.add_argument(
        "--dataset-images-dir",
        default="training_data/dataset/images",
        help="Destination images directory for merged data",
    )
    parser.add_argument(
        "--dataset-labels-dir",
        default="training_data/dataset/labels",
        help="Destination labels directory for merged data",
    )
    parser.add_argument(
        "--weights",
        default="training_data/runs/accepted/weights/best.pt",
        help="Current model weights to fine-tune",
    )
    parser.add_argument(
        "--target-class-id",
        type=int,
        default=7,
        help="Class ID used to rewrite all imported Roboflow labels",
    )
    parser.add_argument(
        "--target-class-name",
        default="soda_can",
        help="Expected class name for --target-class-id in data.yaml",
    )
    parser.add_argument(
        "--runs-project",
        default="training_data/runs",
        help="Ultralytics project directory for run outputs",
    )
    parser.add_argument(
        "--run-name",
        default="accepted",
        help="Run name under the project directory",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of early layers to freeze during fine-tuning",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Disable retrying training on CPU when CUDA allocator errors occur",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    roboflow_root = _resolve_path(args.roboflow_dataset, project_root)
    data_yaml_path = _resolve_path(args.data_yaml, project_root)
    target_images_dir = _resolve_path(args.dataset_images_dir, project_root)
    target_labels_dir = _resolve_path(args.dataset_labels_dir, project_root)
    weights_path = _resolve_path(args.weights, project_root)
    runs_project_dir = _resolve_path(args.runs_project, project_root)

    if not roboflow_root.exists() or not roboflow_root.is_dir():
        raise RuntimeError(f"Roboflow dataset folder not found: {roboflow_root}")
    if not data_yaml_path.exists():
        raise RuntimeError(f"data.yaml not found: {data_yaml_path}")
    if not weights_path.exists():
        raise RuntimeError(f"Model weights not found: {weights_path}")

    class_names = _read_class_names(data_yaml_path)
    target_class_id = int(args.target_class_id)
    target_class_name = str(args.target_class_name).strip().lower()

    if target_class_id not in class_names:
        raise RuntimeError(
            f"Target class ID {target_class_id} is missing from data.yaml names in {data_yaml_path}"
        )
    yaml_class_name = str(class_names[target_class_id]).strip().lower()
    _log(f"data.yaml class[{target_class_id}] = '{class_names[target_class_id]}'")
    if target_class_name and yaml_class_name != target_class_name:
        _log(
            "Warning: target class name mismatch. "
            f"Expected '{target_class_name}' but data.yaml contains '{yaml_class_name}'. "
            "Proceeding with target class ID mapping."
        )

    _log("Merging Roboflow data into existing training_data/dataset")
    (
        imported_label_paths,
        source_class_histogram,
        mapped_annotations,
        invalid_lines,
        empty_labels,
        preview,
    ) = _merge_roboflow_dataset(
        roboflow_root=roboflow_root,
        target_images_dir=target_images_dir,
        target_labels_dir=target_labels_dir,
        target_class_id=target_class_id,
    )

    if source_class_histogram:
        histogram_text = ", ".join(
            f"class {class_id}: {count}"
            for class_id, count in sorted(source_class_histogram.items())
        )
    else:
        histogram_text = "no labeled boxes found"
    _log(f"Source class IDs before remap -> {histogram_text}")
    _log(f"Mapped annotation lines to class {target_class_id}: {mapped_annotations}")
    _log(f"Imported label files: {len(imported_label_paths)} (empty labels: {empty_labels})")
    if invalid_lines > 0:
        _log(f"Warning: skipped {invalid_lines} malformed label lines")

    if preview:
        _log("Mapping preview (source -> target):")
        for line in preview:
            _log(f"  {line}")

    checked_annotations = _verify_imported_labels(imported_label_paths, target_class_id)
    _log(
        "Verification passed: "
        f"all {checked_annotations} non-empty imported annotation lines now use class {target_class_id}"
    )

    runs_project_dir.mkdir(parents=True, exist_ok=True)
    best_path = _train_model(
        weights_path=weights_path,
        data_yaml_path=data_yaml_path,
        runs_project_dir=runs_project_dir,
        run_name=str(args.run_name),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        device=str(args.device),
        lr0=float(args.lr0),
        lrf=float(args.lrf),
        freeze=int(args.freeze),
        patience=int(args.patience),
        allow_cpu_fallback=not bool(args.no_cpu_fallback),
    )
    _log(f"Training complete. best.pt: {best_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[import-train] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

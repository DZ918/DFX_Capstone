#!/usr/bin/env python3
"""Train a local YOLO model from a Kaggle or local detection dataset.

This script is independent from the dashboard flow. It supports:
1) A local dataset folder you already downloaded.
2) A Kaggle dataset URL or owner/name slug (downloaded via Kaggle CLI).

Expected dataset format:
- Preferred: YOLO dataset with a data.yaml file.
- Fallback: image files + YOLO .txt label files that share the same stem.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover - startup dependency guard
    raise RuntimeError(
        "PyYAML is required for local training. Install with: pip install pyyaml"
    ) from exc


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_FILE_CANDIDATES = ("classes.txt", "labels.txt", "obj.names")


def _load_yolo_class():
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - startup dependency guard
        raise RuntimeError(
            "ultralytics is required for training. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def _resolve_path(path_value: str, project_root: Path) -> Path:
    raw = Path(path_value).expanduser()
    return raw if raw.is_absolute() else (project_root / raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local YOLO training from Kaggle or local files (outside dashboard)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", help="Path to local dataset directory")
    source.add_argument(
        "--kaggle-dataset",
        help="Kaggle dataset URL (https://www.kaggle.com/datasets/owner/name) or owner/name",
    )

    parser.add_argument(
        "--kaggle-file",
        default="",
        help="Optional specific file inside the Kaggle dataset",
    )
    parser.add_argument(
        "--download-dir",
        default="training_data/kaggle_downloads",
        help="Where Kaggle datasets are downloaded/extracted",
    )
    parser.add_argument(
        "--training-dir",
        default="training_data",
        help="Project training root (contains dataset/, runs/, class_map.json)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model weights to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Automatic Mixed Precision during training",
    )
    parser.add_argument(
        "--run-name",
        default="accepted",
        help="Run name under training_data/runs (default keeps dashboard compatibility)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split used only when auto-building dataset from image+txt pairs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-names",
        default="",
        help="Comma-separated class names for fallback builder (e.g. canned_drink,water_can)",
    )
    parser.add_argument(
        "--default-class-name",
        default="canned_drink",
        help="Default class name when labels only contain class 0",
    )
    parser.add_argument(
        "--collapse-to-single-class",
        action="store_true",
        help="Rewrite all label class IDs to one class (class 0)",
    )
    parser.add_argument(
        "--single-class-name",
        default="soda_can",
        help="Target class name used with --collapse-to-single-class",
    )
    parser.add_argument(
        "--preserve-existing-classes",
        action="store_true",
        help=(
            "When collapsing to one class, keep existing classes from --model and append "
            "the single class instead of replacing the class list"
        ),
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Disable automatic CPU retry if CUDA allocator/NVML init fails",
    )
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Disable cuDNN before training (useful on constrained Jetson memory setups)",
    )
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


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


def _is_cuda_allocator_issue(message: str) -> bool:
    text = str(message or "").lower()
    markers = (
        "nvml_success",
        "cudacachingallocator",
        "cudaallocatorconfig.cpp",
        "allocator backend parsed at runtime",
        "allocation on device",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda error:",
        "cuda out of memory",
        "driver shutting down",
    )
    return any(marker in text for marker in markers)


def _normalize_kaggle_dataset_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Kaggle dataset value is empty")

    if text.startswith("http://") or text.startswith("https://"):
        match = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", text)
        if not match:
            raise ValueError(
                "Kaggle URL must match https://www.kaggle.com/datasets/<owner>/<dataset>"
            )
        return match.group(1)

    if re.fullmatch(r"[^/\s]+/[^/\s]+", text):
        return text
    raise ValueError("Kaggle dataset must be owner/name or a Kaggle datasets URL")


def _download_kaggle_dataset(
    dataset_id: str,
    download_root: Path,
    kaggle_file: str = "",
    force_redownload: bool = False,
) -> Path:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle, and configure ~/.kaggle/kaggle.json"
        )

    dataset_token = dataset_id.replace("/", "__")
    target_dir = download_root / dataset_token
    if force_redownload and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force_redownload and any(target_dir.iterdir()):
        return target_dir

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(target_dir),
        "--unzip",
    ]
    if kaggle_file:
        command.extend(["-f", kaggle_file])

    subprocess.run(command, check=True)
    return target_dir


def _looks_like_yolo_yaml(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    has_split = "train" in payload or "val" in payload
    has_names = "names" in payload
    return has_split and has_names


def _read_yaml_file(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_yolo_yaml(dataset_root: Path) -> tuple[Path | None, dict | None]:
    preferred = [dataset_root / "data.yaml", dataset_root / "dataset.yaml"]
    candidates: list[Path] = [path for path in preferred if path.exists()]
    if not candidates:
        candidates = sorted(
            [*dataset_root.rglob("*.yaml"), *dataset_root.rglob("*.yml")],
            key=lambda p: (len(p.parts), str(p)),
        )
    for path in candidates:
        payload = _read_yaml_file(path)
        if payload and _looks_like_yolo_yaml(payload):
            return path, payload
    return None, None


def _parse_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        parsed = [str(name).strip() for name in raw_names if str(name).strip()]
        return parsed
    if isinstance(raw_names, dict):
        items = []
        for key, value in raw_names.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            name = str(value).strip()
            if not name:
                continue
            items.append((index, name))
        return [name for _, name in sorted(items, key=lambda item: item[0])]
    return []


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return token.strip("_") or "sample"


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


def _collect_yolo_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(dataset_root.rglob("*.txt")):
        if not label_path.is_file():
            continue
        if label_path.name.lower() in CLASS_FILE_CANDIDATES:
            continue
        image_path = _find_image_for_label(label_path)
        if image_path is None:
            continue
        pairs.append((image_path, label_path))
    return pairs


def _infer_max_class_id(label_paths: list[Path]) -> int:
    max_class_id = 0
    for label_path in label_paths:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except (TypeError, ValueError):
                continue
            if class_id > max_class_id:
                max_class_id = class_id
    return max_class_id


def _find_class_names_file(dataset_root: Path) -> list[str]:
    for file_name in CLASS_FILE_CANDIDATES:
        direct = dataset_root / file_name
        if direct.exists() and direct.is_file():
            lines = [line.strip() for line in direct.read_text(encoding="utf-8").splitlines()]
            names = [line for line in lines if line]
            if names:
                return names
    return []


def _build_class_names(
    args_class_names: str,
    class_file_names: list[str],
    max_class_id: int,
    default_class_name: str,
) -> list[str]:
    if args_class_names:
        names = [token.strip() for token in args_class_names.split(",") if token.strip()]
    elif class_file_names:
        names = class_file_names
    else:
        names = [default_class_name]

    while len(names) <= max_class_id:
        names.append(f"class_{len(names)}")
    return names


def _prepare_clean_dataset_dirs(dataset_dir: Path) -> None:
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        path = dataset_dir / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _rewrite_label_to_single_class(label_source: Path, label_dest: Path, target_class_id: int) -> None:
    """Rewrite one YOLO label file so every annotation uses one class ID."""
    try:
        lines = label_source.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    rewritten_lines: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        rewritten_lines.append(f"{int(target_class_id)} {' '.join(parts[1:])}")

    payload = "\n".join(rewritten_lines)
    if payload:
        payload += "\n"
    label_dest.write_text(payload, encoding="utf-8")


def _build_dataset_from_pairs(
    dataset_root: Path,
    pairs: list[tuple[Path, Path]],
    output_dataset_dir: Path,
    val_split: float,
    seed: int,
    collapse_to_single_class: bool,
    collapse_target_class_id: int,
) -> tuple[int, int]:
    _prepare_clean_dataset_dirs(output_dataset_dir)
    working = list(pairs)
    random.Random(seed).shuffle(working)

    if len(working) <= 1:
        val_count = 0
    else:
        val_ratio = max(0.0, min(0.5, float(val_split)))
        val_count = int(round(len(working) * val_ratio))
        if val_ratio > 0.0 and val_count == 0:
            val_count = 1
        if val_count >= len(working):
            val_count = len(working) - 1

    val_set = set(range(val_count))
    train_count = 0
    for index, (image_path, label_path) in enumerate(working):
        split = "val" if index in val_set else "train"
        image_out = output_dataset_dir / "images" / split
        label_out = output_dataset_dir / "labels" / split
        try:
            rel = image_path.relative_to(dataset_root)
            token_base = _sanitize_token(str(rel.with_suffix(""))).replace("/", "_")
        except ValueError:
            token_base = _sanitize_token(image_path.stem)
        stem = f"{index:06d}_{token_base}"

        image_dest = image_out / f"{stem}{image_path.suffix.lower() or '.jpg'}"
        label_dest = label_out / f"{stem}.txt"
        shutil.copy2(image_path, image_dest)
        if collapse_to_single_class:
            _rewrite_label_to_single_class(
                label_source=label_path,
                label_dest=label_dest,
                target_class_id=int(collapse_target_class_id),
            )
        else:
            shutil.copy2(label_path, label_dest)
        if split == "train":
            train_count += 1

    return train_count, val_count


def _write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    yaml_payload = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path = dataset_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(yaml_payload, handle, sort_keys=False)
    return yaml_path


def _write_class_map(training_dir: Path, class_names: list[str]) -> Path:
    class_map_path = training_dir / "class_map.json"
    payload = {name: index for index, name in enumerate(class_names)}
    with class_map_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return class_map_path


def _resolve_dataset_source(args: argparse.Namespace, project_root: Path) -> Path:
    if args.dataset_dir:
        dataset_root = _resolve_path(args.dataset_dir, project_root)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise RuntimeError(f"Dataset directory does not exist: {dataset_root}")
        return dataset_root

    dataset_id = _normalize_kaggle_dataset_id(args.kaggle_dataset)
    download_root = _resolve_path(args.download_dir, project_root)
    download_root.mkdir(parents=True, exist_ok=True)
    print(f"[train] Downloading Kaggle dataset: {dataset_id}")
    dataset_root = _download_kaggle_dataset(
        dataset_id=dataset_id,
        download_root=download_root,
        kaggle_file=str(args.kaggle_file or "").strip(),
        force_redownload=bool(args.force_redownload),
    )
    print(f"[train] Kaggle dataset ready at: {dataset_root}")
    return dataset_root


def _extract_model_class_names(model_value: str, project_root: Path) -> list[str]:
    """Read class names from a YOLO model file so new classes can be appended safely."""
    YOLO = _load_yolo_class()
    resolved_model_path = _resolve_path(str(model_value), project_root)
    model_ref = str(resolved_model_path) if resolved_model_path.exists() else str(model_value)
    model = YOLO(model_ref)
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        ordered = [str(names[idx]).strip() for idx in sorted(names.keys())]
    elif isinstance(names, list):
        ordered = [str(name).strip() for name in names]
    else:
        ordered = []

    deduped: list[str] = []
    seen: set[str] = set()
    for name in ordered:
        if not name:
            continue
        normalized = name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(name)
    return deduped


def _resolve_data_yaml(
    args: argparse.Namespace,
    dataset_root: Path,
    training_dir: Path,
    preserved_class_names: list[str] | None = None,
) -> tuple[Path, list[str]]:
    existing_yaml, yaml_payload = _find_yolo_yaml(dataset_root)
    if (
        existing_yaml is not None
        and yaml_payload is not None
        and not bool(args.collapse_to_single_class)
    ):
        names = _parse_names(yaml_payload.get("names"))
        if not names:
            fallback_names = [
                token.strip() for token in str(args.class_names or "").split(",") if token.strip()
            ]
            names = fallback_names or [str(args.default_class_name).strip() or "canned_drink"]
        print(f"[train] Found YOLO data file: {existing_yaml}")
        return existing_yaml, names

    pairs = _collect_yolo_pairs(dataset_root)
    if not pairs:
        raise RuntimeError(
            "Could not find YOLO labels. Expected data.yaml or image+txt label pairs in dataset folder."
        )

    output_dataset_dir = training_dir / "dataset"
    label_paths = [label_path for _, label_path in pairs]
    collapse_target_class_id = 0
    if bool(args.collapse_to_single_class):
        requested_name = str(args.single_class_name or "").strip() or "soda_can"
        class_names = [
            str(name).strip()
            for name in (preserved_class_names or [])
            if str(name).strip()
        ]
        if not class_names:
            class_names = [requested_name]
            collapse_target_class_id = 0
        else:
            normalized_to_index = {name.lower(): idx for idx, name in enumerate(class_names)}
            if requested_name.lower() in normalized_to_index:
                collapse_target_class_id = int(normalized_to_index[requested_name.lower()])
            else:
                collapse_target_class_id = len(class_names)
                class_names.append(requested_name)
    else:
        max_class_id = _infer_max_class_id(label_paths)
        class_file_names = _find_class_names_file(dataset_root)
        class_names = _build_class_names(
            args_class_names=str(args.class_names or ""),
            class_file_names=class_file_names,
            max_class_id=max_class_id,
            default_class_name=str(args.default_class_name or "canned_drink"),
        )

    train_count, val_count = _build_dataset_from_pairs(
        dataset_root=dataset_root,
        pairs=pairs,
        output_dataset_dir=output_dataset_dir,
        val_split=float(args.val_split),
        seed=int(args.seed),
        collapse_to_single_class=bool(args.collapse_to_single_class),
        collapse_target_class_id=int(collapse_target_class_id),
    )
    data_yaml = _write_data_yaml(output_dataset_dir, class_names)
    print(
        "[train] Built YOLO dataset from label pairs: "
        f"train={train_count}, val={val_count}, classes={len(class_names)}, "
        f"single_class={bool(args.collapse_to_single_class)}, "
        f"single_class_target_id={int(collapse_target_class_id)}"
    )
    print(f"[train] Generated data file: {data_yaml}")
    return data_yaml, class_names


def _run_training(
    args: argparse.Namespace,
    data_yaml: Path,
    project_root: Path,
    training_dir: Path,
) -> Path:
    running_on_jetson = _is_jetson_linux()
    if running_on_jetson:
        # Must be set before importing torch/ultralytics.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    if bool(args.no_cudnn):
        try:
            import torch

            torch.backends.cudnn.enabled = False
            print("[train] cuDNN disabled via --no-cudnn")
        except Exception as exc:
            print(f"[train] Warning: failed to disable cuDNN: {exc}")
    YOLO = _load_yolo_class()

    model_path = _resolve_path(args.model, project_root)
    if not model_path.exists() and str(args.model).strip() == str(model_path):
        # Allow Ultralytics built-in weights names (e.g. yolov8n.pt)
        model_ref = str(args.model)
    else:
        model_ref = str(model_path)

    runs_dir = training_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_ref)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "project": str(runs_dir),
        "name": str(args.run_name),
        "exist_ok": True,
        "verbose": True,
    }
    if args.batch is not None:
        train_kwargs["batch"] = int(args.batch)
    elif running_on_jetson:
        train_kwargs["batch"] = 4

    if args.workers is not None:
        train_kwargs["workers"] = int(args.workers)
    elif running_on_jetson:
        train_kwargs["workers"] = 1

    if args.amp is not None:
        train_kwargs["amp"] = bool(args.amp)
    elif running_on_jetson:
        # AMP startup checks can trigger NVML allocator assertions on some Jetson stacks.
        train_kwargs["amp"] = False

    if str(args.device or "").strip():
        train_kwargs["device"] = str(args.device).strip()

    if running_on_jetson and bool(args.no_cudnn):
        # Jetson-safe defaults proven to avoid large transient allocations.
        train_kwargs.update(
            {
                "optimizer": "SGD",
                "lr0": 0.001,
                "momentum": 0.9,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "close_mosaic": 0,
                "translate": 0.0,
                "scale": 0.0,
                "degrees": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "plots": False,
                "freeze": 22,
                "val": False,
            }
        )

    print(f"[train] Starting training from model: {model_ref}")
    try:
        result = model.train(**train_kwargs)
    except Exception as exc:
        if args.no_cpu_fallback or not _is_cuda_allocator_issue(str(exc)):
            raise
        retry_kwargs = dict(train_kwargs)
        retry_kwargs["device"] = "cpu"
        retry_kwargs["amp"] = False
        retry_kwargs["workers"] = 0
        if args.batch is None:
            retry_kwargs["batch"] = 2
        print("[train] CUDA allocator/NVML issue detected; retrying training on CPU...")
        result = model.train(**retry_kwargs)

    save_dir = Path(str(getattr(result, "save_dir", runs_dir / str(args.run_name))))
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"Training finished but best.pt not found at: {best_path}")
    return best_path


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    if _is_jetson_linux():
        # Must be set before any torch/ultralytics import for allocator consistency.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    training_dir = _resolve_path(args.training_dir, project_root)
    training_dir.mkdir(parents=True, exist_ok=True)

    preserved_class_names: list[str] = []
    if bool(args.collapse_to_single_class) and bool(args.preserve_existing_classes):
        preserved_class_names = _extract_model_class_names(str(args.model), project_root)
        if preserved_class_names:
            print(f"[train] Preserving {len(preserved_class_names)} existing model classes")
        else:
            print("[train] Warning: no classes were read from --model; proceeding with single-class mode")

    dataset_root = _resolve_dataset_source(args, project_root)
    data_yaml, class_names = _resolve_data_yaml(
        args=args,
        dataset_root=dataset_root,
        training_dir=training_dir,
        preserved_class_names=preserved_class_names,
    )
    class_map_path = _write_class_map(training_dir, class_names)
    print(f"[train] Updated class map: {class_map_path}")

    best_path = _run_training(args, data_yaml, project_root, training_dir)
    print("[train] Training complete")
    print(f"[train] best.pt: {best_path}")
    print("[train] This path is auto-picked by start_gpu.sh/dashboard when run-name is 'accepted'.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[train] Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"[train] {exc}", file=sys.stderr)
        raise SystemExit(1)

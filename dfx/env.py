"""Minimal .env loader for local development."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, *, override: bool = False) -> str | None:
    """Load key=value pairs from a local .env file into os.environ."""
    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if not path.exists() or not path.is_file():
        return None

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return str(path)

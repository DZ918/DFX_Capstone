#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
	PYTHON_BIN="${PYTHON}"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
	PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
	PYTHON_BIN="python3"
fi

GPU_LIB_PATHS="$($PYTHON_BIN - <<'PY'
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

print(":".join(existing))
PY
)"

if [[ -n "$GPU_LIB_PATHS" ]]; then
	export LD_LIBRARY_PATH="${GPU_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "$PYTHON_BIN" "${SCRIPT_DIR}/main.py" "$@"

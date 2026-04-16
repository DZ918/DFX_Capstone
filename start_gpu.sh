#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DFX_PYTHON:-}" ]]; then
	PYTHON_BIN="${DFX_PYTHON}"
elif [[ -x "/home/user/dfx_env/bin/python" ]]; then
	PYTHON_BIN="/home/user/dfx_env/bin/python"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
	PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
	PYTHON_BIN="python3"
fi

# Jetson iGPU has no NVML; use CUDA-native async allocator to avoid NVML asserts
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

MODE="${1:-dashboard}"
if [[ $# -gt 0 ]]; then
	shift
fi

case "$MODE" in
	dashboard)
		exec env PYTHON="$PYTHON_BIN" "${SCRIPT_DIR}/run_dashboard.sh" --cam 0 --model yolov8n.pt "$@" \
			2> >(grep -v 'NvMapMem' >&2)
		;;
	image)
		if [[ $# -lt 1 ]]; then
			echo "Usage: ./start_gpu.sh image <image-path> [additional main.py args]" >&2
			exit 1
		fi
		exec env PYTHON="$PYTHON_BIN" "${SCRIPT_DIR}/run_main.sh" --model yolov8n.pt --image "$1" "${@:2}" \
			2> >(grep -v 'NvMapMem' >&2)
		;;
	help|-h|--help)
		cat <<'EOF'
Usage:
  ./start_gpu.sh
  ./start_gpu.sh dashboard [additional dashboard args]
  ./start_gpu.sh image <image-path> [additional main args]

Defaults:
  - Uses /home/user/dfx_env/bin/python when available
  - Starts dashboard with --cam 0 --model yolov8n.pt

Examples:
  ./start_gpu.sh
  ./start_gpu.sh dashboard --cam 1
  ./start_gpu.sh image ./runs/detect/training_data/runs/smoke/val_batch0_pred.jpg --out /tmp/out.jpg
EOF
		;;
	*)
		echo "Unknown mode: $MODE" >&2
		echo "Run ./start_gpu.sh --help for usage." >&2
		exit 1
		;;
esac
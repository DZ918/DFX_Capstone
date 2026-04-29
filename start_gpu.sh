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

TRAINED_DASHBOARD_MODEL_PATH="${SCRIPT_DIR}/training_data/runs/accepted/weights/best.pt"
BASE_DASHBOARD_MODEL_PATH="${SCRIPT_DIR}/yolov8n.pt"

if [[ -n "${DFX_DASHBOARD_MODEL_PATH:-}" ]]; then
	DASHBOARD_MODEL_PATH="${DFX_DASHBOARD_MODEL_PATH}"
elif [[ -f "$TRAINED_DASHBOARD_MODEL_PATH" ]]; then
	DASHBOARD_MODEL_PATH="$TRAINED_DASHBOARD_MODEL_PATH"
elif [[ -f "$BASE_DASHBOARD_MODEL_PATH" ]]; then
	DASHBOARD_MODEL_PATH="$BASE_DASHBOARD_MODEL_PATH"
else
	DASHBOARD_MODEL_PATH="yolov8n.pt"
fi

DASHBOARD_CONFIDENCE="${DFX_DASHBOARD_CONFIDENCE:-0.35}"

MOTION_FLAG="--motion-enabled"
if [[ -n "${DFX_MOTION_ENABLED:-}" ]]; then
	case "${DFX_MOTION_ENABLED,,}" in
		1|true|yes|on)
			MOTION_FLAG="--motion-enabled"
			;;
		0|false|no|off)
			MOTION_FLAG="--no-motion-enabled"
			;;
		*)
			echo "Invalid DFX_MOTION_ENABLED: ${DFX_MOTION_ENABLED}. Use true/false, on/off, yes/no, or 1/0." >&2
			exit 1
			;;
		esac
fi

MODE="${1:-dashboard}"
if [[ $# -gt 0 ]]; then
	shift
fi

case "$MODE" in
	dashboard)
			exec env PYTHON="$PYTHON_BIN" "${SCRIPT_DIR}/run_dashboard.sh" \
				--cam 0 \
				--model "$DASHBOARD_MODEL_PATH" \
				--conf "$DASHBOARD_CONFIDENCE" \
				--inference-imgsz 320 \
				--max-inference-fps 0 \
					--fps 30 \
				--jpeg-quality 65 \
					"$MOTION_FLAG" \
				"$@" \
				2> >(grep -vE 'NvMapMem|cap_v4l\.cpp|VIDIOC_G_INPUT|obsensor_uvc_stream_channel|video4linux2|cpuinfo: prctl\(PR_SVE_GET_VL\) failed|Created TensorFlow Lite XNNPACK delegate|All log messages before absl::InitializeLog|inference_feedback_manager\.cc|landmark_projection_calculator\.cc' >&2)
		;;
	image)
		if [[ $# -lt 1 ]]; then
			echo "Usage: ./start_gpu.sh image <image-path> [additional main.py args]" >&2
			exit 1
		fi
		exec env PYTHON="$PYTHON_BIN" "${SCRIPT_DIR}/run_main.sh" --model yolov8n.pt --image "$1" "${@:2}" \
			2> >(grep -vE 'NvMapMem|cap_v4l\.cpp|VIDIOC_G_INPUT|obsensor_uvc_stream_channel|video4linux2|cpuinfo: prctl\(PR_SVE_GET_VL\) failed|Created TensorFlow Lite XNNPACK delegate|All log messages before absl::InitializeLog|inference_feedback_manager\.cc|landmark_projection_calculator\.cc' >&2)
		;;
	help|-h|--help)
		cat <<'EOF'
Usage:
  ./start_gpu.sh
  ./start_gpu.sh dashboard [additional dashboard args]
  ./start_gpu.sh image <image-path> [additional main args]

Defaults:
  - Uses /home/user/dfx_env/bin/python when available
	- Starts dashboard with Jetson-safe defaults for imgsz/FPS/JPEG and motion enabled
	- Auto-prefers training_data/runs/accepted/weights/best.pt when it exists
	- Reads DFX_DASHBOARD_MODEL_PATH, DFX_DASHBOARD_CONFIDENCE, and DFX_MOTION_ENABLED when set

Examples:
  ./start_gpu.sh
  ./start_gpu.sh dashboard --cam 1
	DFX_DASHBOARD_MODEL_PATH=snack_model.pt ./start_gpu.sh
	DFX_MOTION_ENABLED=off ./start_gpu.sh
	./start_gpu.sh dashboard --motion-enabled --inference-imgsz 640
  ./start_gpu.sh image ./runs/detect/training_data/runs/smoke/val_batch0_pred.jpg --out /tmp/out.jpg
EOF
		;;
	*)
		echo "Unknown mode: $MODE" >&2
		echo "Run ./start_gpu.sh --help for usage." >&2
		exit 1
		;;
esac
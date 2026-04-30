# DFX_Capstone

python3 dashboard.py \
  --inference-imgsz 960 \
  --max-inference-fps 4 \
  --fps 6 \
  --jpeg-quality 65 \
  --no-motion-enabled

On Jetson, prefer `./run_dashboard.sh` or `./run_main.sh` so the runtime library
path is resolved for the active interpreter automatically.

## Fresh-device setup

Run this once on a new machine:

```bash
python3 install_project.py
```

If you are on Jetson and CUDA-enabled torch is not already installed system-wide,
pass the official NVIDIA wheel that matches your JetPack and Python version:

```bash
python3 install_project.py --torch-wheel <wheel-url-or-path>
```

That script will:

- create a `.venv` virtual environment
- install the Python dependencies from `requirements.txt`
- create the runtime folders used by the app
- create `alerts.json` if it is missing
- download `yolov8n.pt` if it is not already present

After that:

```bash
source .venv/bin/activate
python dashboard.py
```

On Jetson you can also run:

```bash
./run_dashboard.sh
```

## Advanced Detection

The dashboard can run a periodic OpenAI Vision pass in the background while YOLO
continues handling the realtime stream.

This repo now auto-loads a local `.env` file when `dashboard.py` or `main.py`
starts. The fastest setup is:

```bash
cp .env.example .env
```

Then edit `.env` and set your real API key.

If you prefer shell exports instead, these are the same variables:

```bash
export OPENAI_API_KEY=...
export ADVANCED_DETECTION_ENABLED=true
export ADVANCED_DETECTION_INTERVAL_SECONDS=300
export ADVANCED_DETECTION_MODEL=gpt-4.1-mini
export ADVANCED_DETECTION_OUTPUT_DIR=advanced_detections
```

Each run captures one frame from the primary camera plus each active auxiliary
camera, stores the raw frame and JSON sidecar under:

```bash
advanced_detections/YYYY-MM-DD/camera_id/timestamp.jpg
advanced_detections/YYYY-MM-DD/camera_id/timestamp.json
```

If you only want the webcam detector instead of the browser dashboard:

```bash
source .venv/bin/activate
python main.py
```

On Jetson you can also run:

```bash
./run_main.sh
```

## Local Kaggle Training (No Dashboard)

Use `train_local_kaggle.py` when you want to train locally from downloaded files or directly
from a Kaggle dataset link/slug without using the dashboard accept/reject flow.

### 1) Train from local dataset folder

```bash
python train_local_kaggle.py \
  --dataset-dir /path/to/your/dataset \
  --class-names canned_drink
```

If your source dataset has multiple classes but you want one class only (for example all
labels as `soda_can`), use:

```bash
python train_local_kaggle.py \
  --dataset-dir /path/to/your/dataset \
  --collapse-to-single-class \
  --single-class-name soda_can
```

If you want to keep existing classes from your current model and add `soda_can` as a new class,
set your current model path and enable preserving existing classes:

```bash
python train_local_kaggle.py \
  --dataset-dir /path/to/your/dataset \
  --model training_data/runs/accepted/weights/best.pt \
  --collapse-to-single-class \
  --single-class-name soda_can \
  --preserve-existing-classes
```

### 2) Train directly from Kaggle dataset URL or owner/name

```bash
python train_local_kaggle.py \
  --kaggle-dataset https://www.kaggle.com/datasets/<owner>/<dataset> \
  --class-names canned_drink
```

or:

```bash
python train_local_kaggle.py \
  --kaggle-dataset <owner>/<dataset> \
  --class-names canned_drink
```

Kaggle prerequisites:

- Install Kaggle CLI: `pip install kaggle`
- Place API token at `~/.kaggle/kaggle.json`
- Dataset should be YOLO detection format (`data.yaml`) or image+label `.txt` pairs

By default this trains into:

- `training_data/runs/accepted/weights/best.pt`

That is the same path auto-preferred by `start_gpu.sh` and dashboard startup.

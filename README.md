# DFX_Capstone

python3 dashboard.py \
  --inference-imgsz 320 \
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

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

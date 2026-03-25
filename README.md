# DFX_Capstone

python3 dashboard.py \
  --inference-imgsz 320 \
  --max-inference-fps 4 \
  --fps 6 \
  --jpeg-quality 65 \
  --no-motion-enabled

## Fresh-device setup

Run this once on a new machine:

```bash
python3 install_project.py
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

If you only want the webcam detector instead of the browser dashboard:

```bash
source .venv/bin/activate
python main.py
```

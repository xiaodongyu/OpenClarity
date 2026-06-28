# ObjectDetection — Spatial Audio Pipeline

Detects objects in a live webcam stream using YOLOv8n and maps each object's horizontal position to a stereo audio earcon, so a blind/low-vision user can locate objects by sound.

Runs fully on-device (no network) at ≥10 FPS on Ubuntu CPU.

## Requirements

- Ubuntu 22.04+
- Python 3.10+
- Webcam (USB or built-in)
- Audio output device

## Setup

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python-headless sounddevice numpy pytest
```

Pre-download YOLOv8n weights (~6 MB, auto-cached):

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Running the Demo

```bash
bash demo.sh
```

Pass extra flags after `demo.sh`:

```bash
bash demo.sh --visualise          # show bounding box overlay
bash demo.sh --no-audio           # print detections instead of audio
bash demo.sh --conf 0.4           # lower confidence threshold
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MAX_DETECTIONS` | `4` | Max simultaneous earcons (priority filter) |

## Flag Reference

| Flag | Description |
|---|---|
| `--visualise` | Draw bounding boxes with `cv2.imshow` (press `q` to quit) |
| `--no-audio` | Print detections to stdout instead of earcons |
| `--conf FLOAT` | Detection confidence threshold (default: `0.5`) |

## Running Tests

```bash
python3 -m pytest test/ -v
```

## Project Structure

```
ObjectDetection/
├── demo.sh                    # Pre-flight check + launch script
├── src/
│   ├── capture.py             # Webcam frame capture (640×480)
│   ├── detector.py            # YOLOv8n wrapper → list of detection dicts
│   ├── priority_filter.py     # Keep top-N detections by confidence
│   ├── spatial_audio.py       # Stereo earcon synthesis + playback
│   └── pipeline.py            # Main loop + CLI flags
└── test/
    ├── fixtures/
    │   └── indoor_scene.jpg   # Synthetic fixture image
    ├── test_capture.py
    ├── test_detector.py
    ├── test_priority_filter.py
    ├── test_spatial_audio.py
    └── test_pipeline.py
```

## Earcon Frequencies

| Object | Frequency |
|---|---|
| person | 440 Hz (A4) |
| chair | 330 Hz (E4) |
| laptop | 523 Hz (C5) |
| bottle | 392 Hz (G4) |
| door | 261 Hz (C4) |
| cup | 349 Hz (F4) |
| backpack | 466 Hz (Bb4) |
| cell phone | 587 Hz (D5) |
| suitcase | 311 Hz (Eb4) |
| table | 294 Hz (D4) |

Pan position maps to left/right stereo. Loudness scales with bounding box area (proxy for proximity).

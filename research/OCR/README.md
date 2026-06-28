# OCR — On-Device OCR Pipeline

Captures webcam frames, preprocesses them, runs PaddleOCR fully on-device, structures the recognised text, and speaks it via TTS. Zero network dependency — works offline on Ubuntu.

## Requirements

- Ubuntu 22.04+
- Python 3.10+
- Webcam (device `/dev/video0`)

## Setup

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2 espeak
python3 -m venv .venv && source .venv/bin/activate
pip install paddlepaddle paddleocr opencv-python-headless numpy pyttsx3
```

Pre-download model weights (required before first run):

```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
```

Verify the full setup:

```bash
python src/check_setup.py
```

## Usage

### On-demand mode (default)

Press **Space** to capture, preprocess, and read aloud.

```bash
python src/pipeline.py
```

### Continuous mode

Triggers automatically every `CAPTURE_INTERVAL_SEC` seconds (default: 3).

```bash
python src/pipeline.py --continuous
```

### No-TTS (testing / scripting)

Prints recognised text to stdout instead of speaking.

```bash
python src/pipeline.py --no-tts
```

### Language

```bash
python src/pipeline.py --lang ch  # Chinese
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROI_PRESET` | — | `x,y,w,h` to skip interactive ROI selection (e.g. `100,80,400,300`) |
| `CAPTURE_INTERVAL_SEC` | `3` | Seconds between captures in `--continuous` mode |
| `TTS_RATE` | `175` | Speech rate in words per minute |

## Demo

Three test documents are stored in `test/fixtures/demo_docs/` with known ground truth. Run end-to-end accuracy check:

```bash
python src/pipeline.py --no-tts
# Point camera at test/fixtures/demo_docs/doc1.png and press Space
```

## Tests

```bash
pytest test/ -v
```

All 29 tests run offline with no camera or model weights required.

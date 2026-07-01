# Super-Resolution — On-Device Region Zoom

Magnifies a user-selected region of a camera frame with a lightweight
real-time super-resolution neural network (FSRCNN/ESPCN), optionally
combined with adaptive sharpening and Laplacian edge enhancement — replacing
naive bilinear digital zoom, which produces jagged, aliased edges at high
magnification. Built for high-myopia and low-vision users who need to zoom
into small text or objects to see them clearly. Zero network dependency —
works offline on Ubuntu.

## Requirements

- Ubuntu 22.04+
- Python 3.10+
- Webcam (device `/dev/video0`)

## Setup

```bash
sudo apt-get install -y python3-dev python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision opencv-python-headless numpy onnx onnxruntime scikit-image
```

Verify the full setup:

```bash
python -m src.check_setup
```

## Usage

### Zoom pipeline (default: ESPCN + hybrid enhancement)

Press **Space** to capture, upscale, and display the selected zoom region.

```bash
python -m src.pipeline --weights src/weights/espcn_x4.pt
```

### Bilinear + enhancement only (no SR model)

```bash
python -m src.pipeline --no-sr
```

### Choose architecture / scale factor

```bash
python -m src.pipeline --model fsrcnn --scale 2 --weights src/weights/fsrcnn_x2.pt
```

## Training

```bash
python -m src.train --data-dir data/hr_crops --arch espcn --scale 4 \
    --epochs 50 --output src/weights/espcn_x4.pt
```

## Export for edge deployment

```bash
python -m src.export --weights src/weights/espcn_x4.pt --arch espcn --scale 4 \
    --output src/weights/espcn_x4.onnx --quantize
```

## Benchmark vs. bilinear baseline

```bash
python -m src.benchmark --hr-dir test/fixtures/hr_crops --scale 4 \
    --models espcn=src/weights/espcn_x4.pt --report docs/benchmark_report.md
```

## Environment variables

| Variable | Default | Description |
|----------|---------|--------------|
| `ZOOM_PRESET` | — | `x,y,w,h` to skip interactive zoom-region selection (e.g. `100,80,200,150`) |

## Tests

```bash
pytest test/ -v
```

Tests requiring `torch` (`test_sr_model.py`) are skipped automatically if it
isn't installed; all other tests run offline with no camera, GPU, or model
weights required.

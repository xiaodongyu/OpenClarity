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

## Pretrained Baseline Weights

`src/weights/` ships small pretrained checkpoints ported from the public
[FSRCNN](https://github.com/yjn870/FSRCNN-pytorch)/[ESPCN](https://github.com/yjn870/ESPCN-pytorch)
PyTorch implementations (`fsrcnn_x2/x3/x4_pretrained.pt`, `espcn_x3_pretrained.pt`
— ESPCN is only published at x3 upstream). These give a working starting point
before any project-specific fine-tuning (see `docs/finetuning_plan.md`).

To (re-)generate them or pull a specific scale/arch:

```bash
python -m src.convert_pretrained --arch fsrcnn --scale 4 \
    --output src/weights/fsrcnn_x4_pretrained.pt --verify-image path/to/any.jpg
```

`--verify-image` sanity-checks the converted checkpoint against bilinear on a
real image and warns if it doesn't win — see the docstring in
`src/convert_pretrained.py` for an important caveat about bicubic-kernel
sensitivity when doing this comparison yourself.

## Usage

### Zoom pipeline (default: ESPCN + hybrid enhancement)

Press **Space** to capture, upscale, and display the selected zoom region.
ESPCN has no pretrained x4 checkpoint upstream, so pick a matching arch/scale
pair, e.g. FSRCNN at x4:

```bash
python -m src.pipeline --model fsrcnn --scale 4 --weights src/weights/fsrcnn_x4_pretrained.pt
```

### Bilinear + enhancement only (no SR model)

```bash
python -m src.pipeline --no-sr
```

### Choose architecture / scale factor

```bash
python -m src.pipeline --model espcn --scale 3 --weights src/weights/espcn_x3_pretrained.pt
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

Each `--models` entry is `label:arch:scale:weights_path`, so different
labels can share an arch at different scales or share a scale across archs
in a single combined report:

```bash
python -m src.benchmark --hr-dir test/fixtures/medicine_package \
    --models fsrcnn_x2:fsrcnn:2:src/weights/fsrcnn_x2_pretrained.pt \
             fsrcnn_x3:fsrcnn:3:src/weights/fsrcnn_x3_pretrained.pt \
             espcn_x3:espcn:3:src/weights/espcn_x3_pretrained.pt \
             fsrcnn_x4:fsrcnn:4:src/weights/fsrcnn_x4_pretrained.pt \
    --report docs/benchmark_report.md
```

LR is synthesised with PIL's bicubic filter (not OpenCV's) to match the
pretrained checkpoints' training convention — see `docs/algorithm_readme.md`
§2.5.

## HTML eval report

`test/eval_medicine_package.py` runs the same benchmark against the
`test/fixtures/medicine_package/` photos (centre-cropped, since these are
full 12MP phone photos and the actual use case is zooming into one region)
and writes a versioned, self-contained HTML+JSON report — the same
convention used by `research/OCR` and `research/ObjectDetection`'s eval
harnesses:

```bash
python test/eval_medicine_package.py
# -> test/fixtures/medicine_package/eval_<YYYYMMDD>_<NNN>.{html,json}
```

Each card shows the ground-truth crop alongside every method's output
(bilinear, adaptive-sharpen, each pretrained model, and its hybrid variant)
with PSNR/SSIM/latency, and is marked "Improved"/"No improvement" depending
on whether the best SR/hybrid method beat bilinear PSNR.

## Environment variables

| Variable | Default | Description |
|----------|---------|--------------|
| `ZOOM_PRESET` | — | `x,y,w,h` to skip interactive zoom-region selection (e.g. `100,80,200,150`) |

## Tests

```bash
pytest test/ -v
```

Tests requiring `torch` (`test_sr_model.py`, `test_convert_pretrained.py`) are
skipped automatically if it isn't installed; all other tests run offline with
no camera, GPU, network, or model weights required.

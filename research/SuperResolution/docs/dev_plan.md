# Development Plan: On-Device Super-Resolution for Region Zoom

## Goal

High-myopia and low-vision (amblyopia) users often need to "zoom into a specific
region to read text or recognize an object clearly." Naive digital zoom via
bilinear interpolation produces severe jagged edges (aliasing) at high
magnification, making text and object boundaries harder to read, not easier.

This module deploys a lightweight, real-time, on-device super-resolution (SR)
neural network — an optimized **FSRCNN** or **ESPCN** — to upscale a
user-selected region with high fidelity, optionally combined with classical
**adaptive sharpening and Laplacian edge enhancement** for extra crispness on
text and object edges. The full pipeline must run offline, in real time, on
Ubuntu CPU (with an optional GPU/edge-accelerator path for deployment on smart
glasses hardware).

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Camera capture + zoom-region selection | `src/capture.py` |
| 2 | Classical CV baseline (bilinear, adaptive sharpen, Laplacian enhancement) | `src/enhance.py` |
| 3 | Lightweight SR model (FSRCNN / ESPCN) inference wrapper | `src/sr_model.py` |
| 4 | Training / fine-tuning pipeline | `src/train.py` |
| 5 | Model export & on-device optimization (ONNX, quantization) | `src/export.py` |
| 6 | Hybrid pipeline (SR + edge enhancement) | `src/pipeline.py` |
| 7 | Quality/latency benchmark (PSNR, SSIM, FPS vs bilinear) | `src/benchmark.py` |
| 8 | Unit and integration tests | `test/` |

---

## File Structure

```
SuperResolution/
├── docs/
│   └── dev_plan.md
├── src/
│   ├── capture.py       # Webcam capture + interactive zoom-region selection
│   ├── enhance.py        # Bilinear baseline, adaptive sharpening, Laplacian edge enhancement
│   ├── sr_model.py        # FSRCNN/ESPCN architectures + inference wrapper
│   ├── train.py           # Training/fine-tuning loop on DIV2K + synthetic text crops
│   ├── export.py          # ONNX export + INT8 quantization for edge deployment
│   ├── pipeline.py        # Main loop: capture → zoom → SR → enhance → display
│   └── benchmark.py       # PSNR/SSIM/latency comparison harness
└── test/
    ├── test_capture.py
    ├── test_enhance.py
    ├── test_sr_model.py
    ├── test_pipeline.py
    └── fixtures/
        └── text_crops/    # Small labeled crops (signage, labels, product text)
```

---

## Environment Setup

**OS**: Ubuntu 22.04+
**Python**: 3.10+

```bash
sudo apt-get install -y python3-dev python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision opencv-python-headless numpy onnx onnxruntime scikit-image
```

Pretrained FSRCNN/ESPCN weights (or a fine-tuned checkpoint) should be placed
under `src/weights/` and pre-downloaded before the demo:
```bash
python -c "from src.sr_model import SRModel; SRModel(scale=4)"  # triggers weight load/download
```

---

## Phases

### Phase 1 — Capture & Zoom-Region Selection (`src/capture.py`)

**Tasks**
- Open webcam with `cv2.VideoCapture(0)`
- Expose `capture_frame() -> np.ndarray` for a single BGR frame
- Expose `select_zoom_region(frame) -> tuple[int,int,int,int]` via `cv2.selectROI`,
  or a fixed-size box centered on a tap/gaze point for smart-glasses input
- Expose `crop(frame, roi) -> np.ndarray`
- Support `ZOOM_PRESET` env var (e.g. `"100,80,200,150"`) to skip interactive
  selection during demos

**Rationale**: Full-frame SR is wasteful and slow; users only need the specific
region (a label, a sign, a small object) magnified.

**Acceptance criteria**: `test_capture.py` verifies crop dimensions match the
selected region.

---

### Phase 2 — Classical CV Baseline (`src/enhance.py`)

**Tasks**
- `bilinear_upscale(img, scale) -> np.ndarray` — baseline for comparison
  (`cv2.resize(..., interpolation=cv2.INTER_LINEAR)`)
- `adaptive_sharpen(img, amount=1.0) -> np.ndarray` — unsharp mask with a
  locally-adaptive amount based on local variance (avoid over-sharpening flat
  regions / noise amplification)
- `laplacian_edge_enhance(img, strength=0.5) -> np.ndarray` — add a weighted
  Laplacian response back onto the image to boost edge contrast
- `enhance(img) -> np.ndarray` — convenience function chaining sharpen + edge
  enhancement, used as a fallback when the SR model is unavailable and as a
  post-processing step after SR (Phase 6)

**Acceptance criteria**: `test_enhance.py` verifies output dtype/shape and that
a synthetic step-edge image shows increased gradient magnitude after
enhancement without introducing ringing beyond a bounded threshold.

---

### Phase 3 — Lightweight SR Model (`src/sr_model.py`)

**Tasks**
- Implement both architectures behind a common interface, selectable via
  `SR_MODEL` env var (`"fsrcnn"` | `"espcn"`):
  - **FSRCNN**: feature extraction → shrinking → mapping → expanding →
    deconvolution upsampling
  - **ESPCN**: convolutional feature extraction in low-res space + sub-pixel
    (pixel-shuffle) upsampling — cheaper per-pixel cost, preferred for
    real-time edge inference
- Expose `SRModel(scale=4).upscale(img: np.ndarray) -> np.ndarray`
- Load pretrained/fine-tuned weights once at init; run inference in `eval()`
  mode with `torch.no_grad()`
- Log inference latency per call to stderr

**Acceptance criteria**: `test_sr_model.py` runs both models on a fixture crop
and verifies output shape is `scale×` the input and PSNR against a
high-resolution ground truth exceeds the bilinear baseline by a set margin.

---

### Phase 4 — Training / Fine-Tuning (`src/train.py`)

**Tasks**
- Train on DIV2K (general photographic content) with an additional
  synthetic/scraped **text-and-label crop dataset** (signage, product labels,
  medication text) so the model is biased toward sharp glyph edges, which
  matter most for this use case
- Standard SR training recipe: bicubic-downsampled LR/HR pairs, L1 loss (more
  robust to outliers than L2 for edge-heavy content), Adam optimizer
- Track PSNR/SSIM on a held-out validation split each epoch
- Checkpoint best-validation-PSNR model to `src/weights/`

**Acceptance criteria**: Fine-tuned model beats the pretrained baseline PSNR/SSIM
on the text-crop validation split.

---

### Phase 5 — Export & On-Device Optimization (`src/export.py`)

**Tasks**
- Export trained PyTorch model to ONNX (`torch.onnx.export`)
- Apply post-training INT8 quantization (`onnxruntime.quantization`) to reduce
  model size and latency for edge/mobile deployment
- Verify quantized model output stays within an acceptable PSNR delta
  (< 0.5 dB) of the FP32 model
- Save both FP32 and INT8 ONNX artifacts with version tags

**Acceptance criteria**: Quantized ONNX model runs via `onnxruntime.InferenceSession`
and matches the PSNR-delta bound above on the validation set.

---

### Phase 6 — Hybrid Pipeline (`src/pipeline.py`)

**Tasks**
- Main loop: `capture_frame → select/track zoom region → crop → SR upscale →
  enhance (adaptive sharpen + Laplacian) → display`
- Display via `cv2.imshow` (dev) or a framebuffer/HUD output stub for
  smart-glasses integration
- `--no-sr` flag: fall back to `enhance.bilinear_upscale` + `enhance.enhance`
  only, for low-power devices or when the SR model fails to load
- `--scale INT` flag: magnification factor (default: 4)
- `--model {fsrcnn,espcn}` flag

**Demo scenario**: User selects a region containing small text (medicine
label, restaurant menu) or a small object; the system displays a 4× zoomed,
sharp reconstruction on-screen, contrasted live against a bilinear-only zoom
of the same region.

**Acceptance criteria**: End-to-end run on a fixture crop completes
capture→display in real time (target ≥ 15 FPS on CPU for ESPCN at scale 4).

---

### Phase 7 — Benchmark & Quality Evaluation (`src/benchmark.py`)

**Tasks**
- Compare bilinear vs. adaptive-sharpen-only vs. FSRCNN vs. ESPCN vs.
  hybrid (SR + enhancement) on a fixed test set:
  - **Quality**: PSNR, SSIM against ground-truth high-res crops
  - **Speed**: per-frame latency and FPS on CPU
- Generate an HTML/Markdown report with side-by-side crops (mirrors the
  LLM-judge HTML report pattern used in `SceneDescription`)

**Acceptance criteria**: Report clearly shows SR (and hybrid) methods
outperform bilinear in PSNR/SSIM at the target scale factor, with latency
numbers to justify the real-time claim.

---

### Phase 8 — Testing & Demo Prep

**Tasks**
- `test_pipeline.py`: integration test using a fixture crop; verify the
  pipeline produces an upscaled, enhanced output of the expected size
- Pre-download/verify pretrained and fine-tuned weights via a setup check
  script: `python src/check_setup.py`
- Write `README.md` covering setup, env vars, flag reference, and demo run
  command

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SR architecture | ESPCN (default), FSRCNN (alt.) | Sub-pixel convolution upsampling in low-res space is cheapest per-pixel — best fit for real-time edge inference; FSRCNN kept as a higher-quality alternative when compute budget allows |
| Fallback path | Adaptive sharpen + Laplacian edge enhancement | Guarantees a usable (if lower-quality) result when the SR model is unavailable or too slow on a given device |
| Training data | DIV2K + text/label crop dataset | General photographic SR models underperform on small glyph edges, which are the primary use case here |
| Deployment format | ONNX + INT8 quantization | Portable across CPU/edge accelerators without locking into a single runtime; quantization cuts latency/size for on-device deployment |
| Region selection | User-driven zoom ROI (tap/gaze/keypress), not full-frame SR | Full-frame SR at high scale factors is computationally wasteful; users only need one region magnified at a time |
| Loss function | L1 | More robust to outliers than L2, preserves sharp edges better — important for text legibility |

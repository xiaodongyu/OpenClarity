# Algorithm Reference — Super-Resolution

Magnifies a user-selected region of a camera frame with high fidelity so
low-vision users can read small text or identify small objects, replacing
naive bilinear digital zoom (which produces jagged, aliased edges at high
magnification).

```
capture_frame → select_zoom_region → crop → SR upscale (FSRCNN/ESPCN) → adaptive sharpen + Laplacian edge enhance → display
```

If the SR model is unavailable (missing weights, underpowered device), the
pipeline falls back to `bilinear_upscale → enhance`, so the user always gets a
usable — if lower-fidelity — result.

---

## 1. Zoom-Region Selection (`src/capture.py`)

The user selects (via `cv2.selectROI`, or a fixed box around a tap/gaze point
on smart-glasses hardware) the region to magnify. `ZOOM_PRESET` (`x,y,w,h`)
skips interactive selection for scripted demos.

**Rationale**: full-frame SR at high scale factors is computationally
wasteful — at any moment the user only cares about one label, sign, or small
object.

---

## 2. Super-Resolution Model (`src/sr_model.py`)

Two lightweight architectures are implemented behind a common `SRModel`
wrapper, selected via `arch="fsrcnn"|"espcn"`:

### ESPCN (default)

```
Conv(1→64, 5×5) → Tanh → Conv(64→32, 3×3) → Tanh → Conv(32→scale², 3×3) → PixelShuffle(scale)
```

All convolutions run in **low-resolution space**; upsampling only happens in
the final `PixelShuffle` (sub-pixel convolution) layer. This makes ESPCN the
cheapest per-pixel option and the default for real-time on-device inference.

### FSRCNN (alternative, higher quality / higher cost)

```
Conv(1→d, 5×5) [feature extraction]
→ Conv(d→s, 1×1) [shrink]
→ [Conv(s→s, 3×3)] × m [mapping]
→ Conv(s→d, 1×1) [expand]
→ ConvTranspose(d→1, 9×9, stride=scale) [deconvolution upsample]
```

Default hyperparameters: `d=56, s=12, m=4` (the standard FSRCNN-small
configuration). Unlike ESPCN, the final upsampling is a learned deconvolution
rather than pixel-shuffle, at higher compute cost.

### Luma-only processing

Both models operate on the **Y (luma) channel only**:

1. Convert BGR → Y'CbCr, split channels.
2. Run the SR network on Y.
3. Upsample Cb/Cr via bicubic interpolation (`cv2.resize`, `INTER_CUBIC`).
4. Merge and convert back to BGR.

**Rationale**: human vision (and OCR-relevant edge structure) is dominated by
luma; chroma can be upsampled cheaply without a perceptible quality loss. This
roughly triples inference speed vs. running the network on all 3 channels.

**Important**: step 1 uses the **digital, video-range ITU-R BT.601** Y'CbCr
formula (`_bgr_to_ycbcr`/`_ycbcr_to_bgr` in `src/sr_model.py`) —
`Y = 16 + (64.738R + 129.057G + 25.064B)/256`, range ≈16–235 — not OpenCV's
`cv2.COLOR_BGR2YCrCb`, which is a *full-range* transform with a different
scale/offset. This is the convention essentially all SR papers and public
checkpoints (FSRCNN, ESPCN, EDSR, ...) use for the Y channel. Feeding an
externally-pretrained checkpoint OpenCV's full-range Y channel instead
measurably hurts PSNR — it was the first thing to check while validating the
converted checkpoints in `src/convert_pretrained.py` (see below).

---

## 2.5 Pretrained Baseline Weights (`src/convert_pretrained.py`)

Before any project-specific fine-tuning (`docs/finetuning_plan.md`), `src/weights/`
ships FSRCNN (×2/×3/×4) and ESPCN (×3 only — not published upstream at ×4)
checkpoints ported from the public
[yjn870/FSRCNN-pytorch](https://github.com/yjn870/FSRCNN-pytorch) and
[yjn870/ESPCN-pytorch](https://github.com/yjn870/ESPCN-pytorch) repos. Their
architectures are numerically identical to ours (same default hyperparameters,
same layer shapes) but organised into differently named submodules
(`first_part`/`mid_part`/`last_part` vs. our `feature_extraction`/`shrink`/
`mapping`/`expand`/`deconv` and `features`/`to_subpixel`), so `convert_pretrained.py`
remaps state-dict keys 1:1 and loads with `strict=True` to catch any shape/key
mismatch before saving.

### Known pitfall: bicubic-kernel mismatch when verifying

These checkpoints were trained on LR images generated with **PIL's `BICUBIC`
filter**, not OpenCV's. FSRCNN/ESPCN — like most classical, non-blind SR
nets — are sensitive to the *exact* downsampling kernel used to create their
training LR data. Synthetically downsampling a test image with
`cv2.INTER_CUBIC` (what `src/benchmark.py` and `src/train.py` use internally —
fine there, since each is self-consistent end-to-end) and feeding that into an
*externally pretrained* checkpoint makes it score **worse than bilinear**, even
though the checkpoint itself is correct — a kernel-mismatch artifact, not a
model defect. Measured on a 256×256 test image (`fsrcnn x3`): cv2-bicubic LR
gave 20.56 dB (bilinear: 21.89 dB); PIL-bicubic LR gave 26.37 dB (bilinear:
21.96 dB) — the same checkpoint, only the LR-generation kernel changed.

`convert_pretrained.py`'s `verify_against_bilinear()` therefore generates its
sanity-check LR with PIL's bicubic filter specifically. Any future fine-tuning
must keep the LR-generation kernel consistent between whatever data a
checkpoint is trained on and whatever it's evaluated on.

---

## 3. Classical CV Enhancement (`src/enhance.py`)

Used both as the no-SR fallback and as a post-processing step after SR
(hybrid mode), to further crisp text/object edges.

### Adaptive sharpening

Standard unsharp masking, but the sharpening **amount is locally weighted by
image variance**:

```python
detail = img - gaussian_blur(img, sigma=3)
weight = amount * (local_variance / max(local_variance))
sharpened = img + weight * detail
```

`local_variance` is computed over a 9×9 window (`cv2.blur` on `img` and
`img²`). Flat, low-detail regions (weight ≈ 0) are left untouched — avoiding
noise amplification — while high-contrast regions (text strokes, object
edges) get close to the full sharpening amount.

### Laplacian edge enhancement

```python
laplacian = cv2.Laplacian(gray, ksize=3)
enhanced = img - strength * laplacian
```

Subtracting the Laplacian response boosts local contrast at edges (a
second-derivative sharpening term), applied after adaptive sharpening
(`enhance()` chains both).

---

## 4. Training (`src/train.py`)

- **Data**: DIV2K HR crops + a text/label crop dataset (signage, product
  labels), so the model is biased toward sharp glyph edges rather than
  general photographic content.
- **LR generation**: bicubic downsampling of random HR patches by `scale`
  (standard FSRCNN/ESPCN recipe) — no external LR dataset needed.
- **Loss**: L1 (more robust to outliers than L2; better edge preservation
  than L2, which tends to blur high-frequency detail).
- **Model selection**: best-validation-PSNR checkpoint is kept.

---

## 5. Export & On-Device Optimization (`src/export.py`)

1. `torch.onnx.export` with dynamic height/width axes (so one exported model
   handles any zoom-region size).
2. Optional INT8 post-training dynamic quantization
   (`onnxruntime.quantization.quantize_dynamic`) for edge/mobile deployment —
   reduces model size and CPU latency.
3. `max_psnr_delta` gates the quantized model against a validation crop set;
   deployment should only proceed if the INT8 PSNR drop stays under ~0.5 dB.

---

## 6. Evaluation (`src/benchmark.py`)

Compares, on held-out HR images (with LR generated by **PIL-bicubic**
downsampling — see §2.5 for why not cv2):

| Method | Description |
|---|---|
| `bilinear_x{scale}` | Baseline naive zoom |
| `adaptive_sharpen_x{scale}` | Bilinear + adaptive sharpen (no SR model) |
| `<label>` | SR model alone, as specified by a `label:arch:scale:weights_path` entry |
| `<label>_hybrid` | SR model output + adaptive sharpen + Laplacian enhance |

Models are specified per-scale (`ModelSpec`), so a single run can combine
different archs at the same scale (e.g. `fsrcnn_x3` vs. `espcn_x3`) or the
same arch at different scales (e.g. `fsrcnn_x2/x3/x4`) — results are grouped
by `(scale, method)` throughout, since PSNR/SSIM are only comparable within
the same scale.

Metrics per method (averaged across the image set, grouped by scale):
- **PSNR** (dB) — reconstruction fidelity vs. ground truth
- **SSIM** — perceptual structural similarity
- **Latency** (ms) — per-image inference/processing time

Results are written to a Markdown report (`write_report`) ranked by PSNR
within each scale section.

### HTML eval harness (`test/eval_medicine_package.py`)

Runs the same evaluation against `test/fixtures/medicine_package/` (three
photos, centre-cropped to 800×600 since these are full 12MP phone photos and
the actual use case zooms into one region, not the whole frame) and writes a
versioned, self-contained HTML+JSON report — the same convention as
`research/OCR/test/eval_fixtures.py` and
`research/ObjectDetection/test/eval_home_scene.py` (base64-embedded images,
dark header with commit info, `eval_<date>_<seq>` run IDs, green/orange
pass-count summary banner). Each card's verdict is "Improved" if the best
SR/hybrid method beats bilinear PSNR for that image×scale.

**First baseline run** (`eval_20260701_001`, all 4 pretrained checkpoints):
**9/9 image×scale combinations improved over bilinear** — e.g. `fsrcnn_x2`
35.7–38.5 dB vs. bilinear 31.9–37.3 dB across the three fixture photos.

**Finding — hybrid mode currently hurts, doesn't help, on top of a good SR
reconstruction.** `<label>_hybrid` (SR output + `enhance()`) scored **7–14 dB
lower** than the SR model alone on every single card in this run, e.g.
`fsrcnn_x2` 35.70 dB vs. `fsrcnn_x2_hybrid` 21.77 dB. `enhance()`'s adaptive
sharpen + Laplacian boost were tuned against a *blurry* bilinear-only input
(its intended no-SR fallback role); applying the same fixed strength on top
of an already-sharp SR reconstruction over-sharpens and introduces
ringing/overshoot large enough to tank PSNR. The evaluation's best-method
selection already accounts for this (hybrid never wins the per-card
comparison in this run, so the 9/9 verdict isn't masking it), but hybrid mode
should not be treated as strictly better than the raw SR model until
`enhance()`'s strength is made conditional on whether its input already went
through SR — worth a follow-up before Phase 4/5 fine-tuning treats "hybrid"
as the default deployment path.

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default SR architecture | ESPCN | Cheapest per-pixel cost (all convs in LR space); best fit for real-time edge inference |
| Alternative architecture | FSRCNN | Higher quality when compute budget allows; kept behind the same interface |
| Color handling | Y-channel only, bicubic chroma | ~3× faster than 3-channel inference with no perceptible quality loss |
| Y'CbCr convention | Digital/video-range BT.601 (not OpenCV's full-range `cv2.COLOR_BGR2YCrCb`) | Matches the convention used by the FSRCNN/ESPCN papers and public checkpoints; required for pretrained weights to perform correctly |
| Baseline weights | Ported from yjn870's public FSRCNN/ESPCN PyTorch checkpoints | Working starting point before project-specific fine-tuning; architectures match ours exactly, only submodule naming differs |
| Training data | DIV2K + text/label crops | General SR models underperform on small glyph edges, which matter most for this use case |
| Loss function | L1 | More robust to outliers, preserves sharp edges better than L2 |
| Deployment format | ONNX (+ optional INT8 quantization) | Portable across CPU/edge runtimes; quantization cuts latency/size without hard-locking a single inference engine |
| Fallback path | Bilinear + adaptive sharpen/Laplacian enhance | Guarantees a usable result when the SR model is missing or too slow for the device |
| Region selection | User-driven zoom ROI, not full-frame SR | Full-frame SR at high scale factors is wasted compute; only one region is read at a time |

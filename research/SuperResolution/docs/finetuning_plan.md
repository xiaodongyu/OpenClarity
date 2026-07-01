# Fine-Tuning Plan: SR Model for Text/Label Legibility

## Why Fine-Tuning Is Necessary

`SRModel` (`src/sr_model.py`) currently only builds with **randomly-initialized**
weights — there is no trained checkpoint in this repo yet. Public pretrained
FSRCNN/ESPCN weights exist (yjn870, Lornatang, Nhat-Thanh PyTorch ports;
OpenCV's `dnn_superres` `.pb` models), but all of them share the same gap:

- They are trained on **generic natural-photo datasets** (T91, General100,
  or DIV2K) — landscapes, faces, textures. These optimize for photographic
  fidelity, not for **sharp glyph edges** at small font sizes.
- This project's actual use case — a high-myopia/low-vision user zooming
  into a medicine label, sign, or menu — is dominated by thin strokes,
  serifs, and high-contrast text-on-background edges, which generic SR
  training underweights relative to photographic textures.
- Layer *names* also differ between public checkpoints and our
  `_build_fsrcnn`/`_build_espcn` modules (shapes match, keys don't), so even
  reusing a pretrained checkpoint requires a remapping step before it's
  loadable at all.

Net effect: without fine-tuning (or at least remapped pretrained weights +
fine-tuning), the model in this repo cannot yet outperform bilinear zoom on
the target use case — it needs to be trained.

---

## What Fine-Tuning Requires

| Requirement | Detail |
|---|---|
| **Starting weights** | Either (a) train from scratch on DIV2K, or (b) port a public pretrained checkpoint (yjn870/Lornatang) via a key-remapping script and fine-tune from there. (b) is faster to converge — reuses low-level filters (edges/gradients) already learned. |
| **Domain data** | A text/label crop dataset: photos of signage, medicine labels, product packaging, menus, price tags, book spines — the actual content this tool is used on. |
| **General data** | DIV2K (or T91/General100) HR crops, mixed in alongside the text crops, so the model doesn't overfit to a narrow domain and lose general reconstruction quality (e.g. skin tones, object edges also matter — the camera sees more than just labels). |
| **Data volume** | A few hundred to ~1–2k text/label crops is a reasonable starting target — small SR nets (ESPCN/FSRCNN) don't need DIV2K-scale volume, but variety (fonts, lighting, angles, surface types — flat vs. curved) matters more than raw count. |
| **Augmentation** | Random crop, horizontal flip, mild rotation (±5°), brightness/contrast jitter — needed because the domain dataset will be small; without augmentation the model will overfit to specific lighting/framing. |
| **Compute** | CPU-feasible given the model sizes (ESPCN ~20k params, FSRCNN-small ~12k params) but a GPU (even a laptop GPU) will cut iteration time from hours to minutes per run. |
| **Evaluation ground truth** | Held-out HR crops (not used in training) for `src/benchmark.py`'s PSNR/SSIM comparison. |
| **Legibility proxy metric** | PSNR/SSIM don't directly measure "can a person read this text." A legibility proxy — running the existing OCR project's PaddleOCR wrapper (`research/OCR/src/ocr_engine.py`) on SR output vs. bilinear output and comparing recognition confidence/CER — is a more decision-relevant signal for this specific use case. |
| **Quantization re-validation** | After fine-tuning, re-run `src/export.py`'s `max_psnr_delta` check — INT8 quantization error tolerance was set against generic content; must reconfirm it holds on text crops, since quantization tends to hurt fine, high-frequency detail (exactly what text edges are) more than smooth content. |

---

## Phases

### Phase 1 — Baseline Weights

**Tasks**
- Write `src/convert_pretrained.py`: downloads a public pretrained checkpoint
  (e.g. yjn870's ESPCN/FSRCNN `.pth`), remaps its state-dict keys onto our
  `_build_espcn`/`_build_fsrcnn` module structure, and saves it in a format
  `SRModel(weights_path=...)` can load.
- Verify the remapped checkpoint reproduces the original paper's reported
  PSNR on a small Set5/Set14-style sanity check (or, at minimum, clearly
  outperforms bilinear on a handful of test images) before treating it as a
  valid starting point.

**Acceptance criteria**: remapped model loads without shape/key errors and
PSNR on a quick sanity set exceeds bilinear.

---

### Phase 2 — Domain Dataset Collection

**Tasks**
- Collect real photos: medicine labels, product packaging, signage, menus,
  price tags, book spines — ideally shot with the same camera/lens class
  intended for deployment (webcam or smart-glasses camera), across varied
  lighting and distances.
- Supplement with existing public OCR datasets for volume/variety if needed
  (e.g. ICDAR, SVT, COCO-Text, TextOCR) — crop to the same style of tight,
  text-bearing regions used at inference time.
- Curate: discard blurry/motion-blurred source images (fine-tuning on blurry
  "ground truth" would teach the model to reproduce blur).
- Split into train/val (~90/10), held out at the crop level (not just image
  level) to avoid leakage between near-duplicate crops from the same photo.

**Acceptance criteria**: `data/text_crops/{train,val}/` populated; manual spot
check confirms crops are sharp, in-focus, and representative of real usage.

---

### Phase 3 — Data Pipeline: Mixed-Domain Sampling

**Tasks**
- Extend `HRCropDataset` (`src/train.py`) to accept **two source
  directories** (general + domain) with a configurable sampling ratio
  (e.g. `--domain-ratio 0.5`), so each training batch mixes DIV2K and
  text-crop patches rather than training on one then the other.
- Add the augmentation transforms (flip, small rotation, brightness/contrast
  jitter) described above, applied only to the training split.

**Acceptance criteria**: a training epoch draws from both sources in roughly
the configured ratio; augmented samples visually verified (dump a few
augmented crops to check they're not corrupted).

---

### Phase 4 — Fine-Tuning Run

**Tasks**
- Fine-tune from the Phase 1 remapped checkpoint (lower learning rate than
  training-from-scratch, e.g. `1e-4` vs. `1e-3`) using the Phase 3 mixed
  dataloader.
- Track validation PSNR/SSIM **separately** for the general-domain val split
  and the text-crop val split each epoch — catches the case where text
  performance improves while general performance regresses.
- Early-stop / checkpoint on best text-crop validation PSNR (the metric that
  matters for this use case), not overall/general PSNR.

**Acceptance criteria**: fine-tuned checkpoint's text-crop val PSNR/SSIM
exceeds both (a) the Phase 1 baseline checkpoint and (b) bilinear, without a
large regression (> 1 dB) on the general-domain val split.

---

### Phase 5 — Legibility Evaluation

**Tasks**
- Extend `src/benchmark.py` (or add a sibling script) to run the existing
  PaddleOCR wrapper from `research/OCR/src/ocr_engine.py` on each method's
  output (bilinear, adaptive-sharpen, SR, hybrid) for a set of text crops
  with known ground-truth text.
- Compute CER (reusing `research/OCR`'s evaluation approach) per method as a
  legibility proxy, in addition to PSNR/SSIM.
- Confirm the fine-tuned model (and hybrid post-processing) reduces CER
  relative to bilinear zoom — this is the metric that actually reflects "can
  the user read the zoomed text," which PSNR/SSIM only approximate.

**Acceptance criteria**: fine-tuned SR (or hybrid) achieves lower mean CER
than bilinear across the text-crop benchmark set.

---

### Phase 6 — Export & Re-Quantization

**Tasks**
- Re-run `src/export.py` on the fine-tuned checkpoint.
- Re-check `max_psnr_delta` (INT8 vs FP32) specifically on the text-crop val
  set — text/edge content is more quantization-sensitive than smooth
  content, so the delta bound established during initial development must
  be re-validated, not assumed to still hold.

**Acceptance criteria**: INT8 PSNR delta on text crops stays within the
project's tolerance (~0.5 dB); if not, fall back to FP32 for text-heavy
deployment or widen the quantization calibration set.

---

### Phase 7 — Integration

**Tasks**
- Commit the fine-tuned weights (or a documented download step, if too large
  for the repo) as the default path referenced in `README.md` and
  `src/pipeline.py`'s `--weights` default.
- Update `docs/algorithm_readme.md`'s training section with the actual
  dataset composition and final benchmark numbers once available.

---

## Key Open Decisions

| Decision | Options | Notes |
|---|---|---|
| Start from scratch vs. remapped pretrained weights | Remapped pretrained (recommended) | Faster convergence; low-level filters transfer from generic photo SR |
| Domain-crop source | Self-collected vs. public OCR datasets vs. both (recommended) | Self-collected matches deployment camera best; public datasets add volume/variety faster |
| Sampling ratio (domain vs. general) | Start at 50/50, tune based on Phase 4 results | Too domain-heavy risks losing general reconstruction quality on non-text content in frame |
| Model selection metric | Text-crop val PSNR (not overall PSNR) | Matches the actual deployment priority (legibility) established in `docs/algorithm_readme.md` |

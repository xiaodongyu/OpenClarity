# Algorithm Reference — OCR

Captures a camera frame, extracts text using PaddleOCR, and speaks the result
via TTS.

```
capture_frame → pick_orientation → PaddleOCR → structure_text → TTS
```

---

## 1. Orientation Detection

Before running OCR the image is rotated so that text rows are horizontal.
Orientation detection runs in two phases.

### Phase 1 — Projection-profile variance (`src/preprocess.py`)

Scores each cardinal orientation (0°, 90°, 180°, 270°) by the **variance of
the horizontal row-sum profile** on a binarised image.

When text rows are horizontal the binary image has alternating bands of high
ink density (text rows) and low ink density (gaps), producing high row-sum
variance. At 90° the same ink is spread across columns; the profile is nearly
flat and variance is low.

**Steps:**

1. Convert to grayscale.
2. Downsample to ≤ 600 px on the longest side (speed).
3. Gaussian blur (5×5) + Otsu threshold → binary image.
4. For each angle rotate the binary image with `cv2.rotate`, then:
   ```python
   profile = np.sum(candidate, axis=1, dtype=np.float64)
   var = float(np.var(profile))
   ```
5. Return the angle with the highest variance as the initial best.

### Phase 2 — OCR-confidence tiebreaker (`src/ocr_engine.py`)

Variance is mathematically equal for 0° vs 180° (reversing rows doesn't change
variance) and for 90° vs 270°. Whenever the top two angles are 180° apart and
the relative variance gap is < 2%, the algorithm falls back to running
downsampled OCR on both candidates and comparing total recognition confidence.

| Tied pair   | OCR run with      | Rationale |
|-------------|-------------------|-----------|
| 0° / 180°   | `cls=False`       | Inverted text is unreadable; its confidence sum is near zero, so the upright side wins cleanly. |
| 90° / 270°  | `cls=True`        | The angle classifier is needed to recover text that appears upside-down after rotation (e.g. a horizontal banner visible at both rotations); without it one side would always win regardless of which physical face of the object is showing. |

The public API for the combined algorithm is **`pick_orientation(img)`** in
`src/ocr_engine.py`. Both `recognize()` and the evaluation annotator call this
function so the rotation applied to the image and the rotation applied to the
bounding boxes are always the same.

**Known limitation:** if a subject is photographed with its back face showing
more legible text than the front face (e.g. back-cover blurb vs decorative
title font), the OCR-confidence tiebreaker will prefer the back-cover
orientation. Fixing this would require a model that reads decorative fonts or
external context (e.g. the user pointing the camera at the front cover).

---

## 2. PaddleOCR Recognition

PaddleOCR (v2.9, engine: `paddlepaddle==2.6.2`) runs a three-stage pipeline
internally:

1. **Text detection** (DB-MobileNetV3) — finds bounding polygons around text
   regions.
2. **Angle classifier** — predicts 0° vs 180° orientation per word crop.
3. **Text recognition** (CRNN) — reads the character sequence from each crop.

The wrapper (`src/ocr_engine.py: recognize`) filters results by a confidence
threshold (default 0.6) and returns a list of token dicts:

```python
{"text": str, "confidence": float, "bbox": list[list[float]]}
```

---

## 3. Reading-Order Reconstruction

Tokens are sorted into reading order by `src/text_formatter.py: structure_text`:

1. Sort tokens by `(min_y, min_x)` of their bounding box.
2. Group tokens into lines using vertical overlap: two tokens belong to the same
   line if their bounding boxes overlap by more than −10 px vertically.
3. Within each line, sort tokens left-to-right by `min_x`.
4. Join tokens per line with spaces; join lines with newlines.

### Known limitation — curved surfaces

On cylindrical packaging (e.g. medicine tubes), all text lines wrap around the
surface and appear at nearly the same y-coordinate in the 2D projection. The
top-to-bottom, left-to-right sort cannot recover the correct reading order in
this case. Fixing it would require either a 3D unwrap step or a learned
reading-order model.

---

## 4. Text-to-Speech

A daemon thread (`src/tts.py`) drains a priority queue. Each call to `speak()`
enqueues the text. If a new item arrives while speech is in progress, the queue
is drained and only the latest item is spoken — stale descriptions are skipped.

---

## 5. Evaluation

The eval script (`test/eval_fixtures.py`) supports two evaluation modes
selected automatically from the ground truth format.

### Mode A — Character Error Rate (CER)

Used when the ground truth line contains no `|` character (flat, ordered text).

```
CER = edit_distance(reference, hypothesis) / len(reference)
```

The **edit distance** is the minimum number of single-character insertions,
deletions, and substitutions to transform the hypothesis into the reference,
computed with standard O(|ref| × |hyp|) dynamic programming.

CER can exceed 100% when the hypothesis is longer than the reference
(hallucinated tokens). It is computed case-insensitively.

Pass threshold: CER < 30%.

### Mode B — Key-phrase Recall

Used when the ground truth line contains `|`-delimited phrases (curved surfaces
or text whose reading order is unspecified):

```
IMG_4255.JPG: Walgreens | Sleep Aid
```

Each phrase is looked up in the recognised text using a three-tier fuzzy match:

1. **Exact substring** (spaces preserved).
2. **No-space substring** — handles merged tokens such as `STATEOFFEAR`.
3. **Word overlap** — ≥ 75% of the phrase's words appear in the recognised word
   set (order-insensitive).

All comparisons are case-insensitive after stripping non-alphanumeric characters.

```
recall = (phrases found) / (total phrases)
```

Pass threshold: recall ≥ 70%.

### Summary metric

The HTML/JSON report shows **N / Total passed** per fixture (count of images
that individually pass their threshold). Mean CER is retained in the JSON for
post-hoc analysis but is no longer the primary summary metric.

### Benchmark fixtures

| Fixture set         | Images | Mode       | Surface type        | Notes |
|---------------------|--------|------------|---------------------|-------|
| `demo_docs/`        | 3      | CER        | Flat printed pages  | Baseline; controlled lighting |
| `book_cover/`       | 3      | Key-phrase | Flat glossy cover   | Variable font sizes; decorative titles |
| `medicine_package/` | 3      | Key-phrase | Flat box + cylindrical tube | Curved surface; reading order unspecified |

Ground truth files use the format `filename: expected text`, one line per image.

---

## 6. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| OCR engine | PaddleOCR 2.9 / PaddlePaddle 2.6.2 | v3.x incompatible with this CPU's oneDNN |
| Orientation detection phase 1 | Projection-profile variance | No ML model needed; works on any binary image |
| Orientation detection phase 2 | OCR-confidence tiebreaker | Resolves the 0°/180° and 90°/270° variance ties that phase 1 cannot distinguish |
| Tiebreaker cls flag | `cls=False` for 0°/180°, `cls=True` for 90°/270° | For horizontal pairs the classifier is needed to recover upside-down incidental text; for vertical pairs inverted text scores near-zero without it |
| OCR confidence threshold | 0.6 | Empirically reduces noise on medicine labels |
| Evaluation mode A | CER | Fairer than WER for partial OCR reads and short labels on flat, ordered surfaces |
| Evaluation mode B | Key-phrase recall | Order-insensitive; appropriate for curved/scattered text where reading order reconstruction fails |
| TTS interruption | Daemon thread + priority queue; drain on wakeup | Avoids stale descriptions when scene changes |

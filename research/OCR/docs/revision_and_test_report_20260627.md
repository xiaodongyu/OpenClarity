# OCR — Revision and Test Report — 2026-06-27

Summary of improvements made and evaluation results recorded during this session.

---

## Improvements

### 1. Fix bounding-box annotation misalignment
**Commit:** `90bb422`  
**File:** `src/ocr_engine.py`, `test/eval_fixtures.py`

`_annotate_image_b64` was calling `correct_orientation()` (variance-only) from
`preprocess.py` while `recognize()` used the full tiebreaker logic in
`ocr_engine.py`. The two could choose different angles, causing bounding boxes
to be drawn on the wrong rotation of the image.

**Fix:** Extracted `pick_orientation(img) -> int` as a public function in
`ocr_engine.py` that encapsulates both the variance phase and the OCR-confidence
tiebreaker. Both `recognize()` and the HTML annotator now call the same function,
guaranteeing the image shown and the coordinates drawn always match.

---

### 2. Add key-phrase recall evaluation mode
**Commit:** `f3767ff`  
**Files:** `test/eval_fixtures.py`, `test/fixtures/book_cover/ground_truth.txt`,
`test/fixtures/medicine_package/ground_truth.txt`

CER (character error rate) is order-sensitive and penalises the correct tokens
when reading order is scrambled — exactly what happens on curved packaging or
when a photo shows scattered text. For these fixtures, presence of key phrases
matters more than sequence.

**New Mode B — Key-phrase recall:**
- Ground truth lines use `|`-delimited phrases:
  `IMG_4255.JPG: Walgreens | Sleep Aid`
- Each phrase is looked up in the recognised text with a three-tier fuzzy match:
  1. Exact substring (spaces preserved).
  2. No-space substring — handles merged tokens such as `STATEOFFEAR`.
  3. Word overlap — ≥ 75% of the phrase's words appear in the recognised word set (order-insensitive).
- Pass threshold: **recall ≥ 70%** (≥ 70% of phrases found).
- Mode is selected automatically: `|` in the ground truth line → Mode B; otherwise Mode A (CER < 30%).

Ground truth files for `book_cover/` and `medicine_package/` were updated to
the `|`-delimited format.

---

### 3. Fix 90°/270° orientation tiebreaker
**Commit:** `a7baa35`  
**File:** `src/ocr_engine.py`

The OCR-confidence tiebreaker was running with `cls=False` for all tied pairs.
This works for 0°/180° (inverted text scores near-zero without the classifier),
but fails for 90°/270°: a horizontal banner (e.g. "THE NEW YORK TIMES
BESTSELLER") appears at both rotations, and without the angle classifier the
version where the banner happens to be upright always wins — regardless of
which physical face of the subject is showing.

**Fix:** `cls=True` for 90°/270° pairs so the angle classifier can recover text
that appears upside-down after rotation, giving a fairer confidence signal.

**Result:** IMG_4259 (Robin Cook VECTOR) orientation improved in some cases;
IMG_4260 (David Morrell DOUBLE IMAGE) still picks the back cover because it
contains far more legible blurb text than the decorative title font on the front
— a fundamental limitation of OCR-confidence-based tiebreaking.

---

### 4. Update HTML report summary metric
**Commit:** `8f10412`  
**File:** `test/eval_fixtures.py`

The HTML report header previously showed mean CER as the primary metric
(e.g. "221.3% mean CER"), which is misleading when most images use key-phrase
recall mode. Replaced with **pass/fail count** (e.g. "3 / 3 test cases passed").
Removed the CER band legend panel. Per-image badges still show the relevant
metric (Recall % or CER %) depending on the mode for that image.

---

### 5. Update algorithm README
**Commit:** `8f10412`  
**File:** `docs/algorithm_readme.md`

Rewrote to document all of the above:
- Orientation detection split into Phase 1 (variance) and Phase 2 (tiebreaker),
  including the `cls` flag rationale table and the back-cover limitation.
- Evaluation section split into Mode A (CER) and Mode B (key-phrase recall)
  with thresholds, fuzzy-match tiers, and updated benchmark fixture table.
- Design decisions table expanded with the tiebreaker and both eval modes.

---

## Test Results

Eval artifact commit: `8870350` — tested at source commit `8f10412`.

### book_cover (Mode B — Key-phrase recall, threshold ≥ 70%)

| Image | Expected phrases | Phrases found | Recall | Verdict |
|-------|-----------------|---------------|--------|---------|
| IMG_4258.JPG | MICHAEL CRICHTON, STATE OF FEAR | both | 100% | ✓ PASS |
| IMG_4259.JPG | ROBIN COOK, VECTOR | none | 0% | ✗ FAIL |
| IMG_4260.JPG | DAVID MORRELL, DOUBLE IMAGE | DAVID MORRELL only | 50% | ✗ FAIL |

**1 / 3 passed**

**IMG_4259 failure:** Only "THE NEW YORK TIMES BESTSELLER" was detected. The
orientation tiebreaker chose the spine/banner side of the book rather than the
cover face showing the author name and title. Both cls fixes applied but the
banner side still scores higher.

**IMG_4260 failure:** The back cover is visible. Its blurb text ("INVENTIVE, AND
GRIPPING. LEAVES YOU DAZZLED." — Dean Koontz; "WARNER VISION") produces higher
OCR confidence than the decorative front-cover font for "DOUBLE IMAGE".
DAVID MORRELL is found (also printed on the back cover), but the title is not.
This is a known algorithmic limitation — OCR confidence cannot distinguish a
decorative title font from a blurb.

---

### medicine_package (Mode B — Key-phrase recall, threshold ≥ 70%)

| Image | Expected phrases | Phrases found | Recall | Verdict |
|-------|-----------------|---------------|--------|---------|
| IMG_4255.JPG | Walgreens, Sleep Aid | both | 100% | ✓ PASS |
| IMG_4256.JPG | Triamcinolone Acetonide, Ointment | both | 100% | ✓ PASS |
| IMG_4257.JPG | Clobetasol Propionate, Ointment USP | both | 100% | ✓ PASS |

**3 / 3 passed**

All three medicine packages are correctly identified despite reading-order
scrambling due to cylindrical surfaces. The key-phrase recall mode handles
merged tokens ("Ointment)" → matches "Ointment USP") and reordered output.

---

## Known Limitations

| Issue | Root cause | Workaround / fix path |
|-------|-----------|----------------------|
| IMG_4259 wrong orientation | OCR confidence on banner side > title side | Better photo (front face centred); or VLM-based orientation classifier |
| IMG_4260 back cover preferred | Back-cover blurb is more machine-readable than decorative title font | Better photo; or prompt user to point camera at front cover |
| Reading order on curved surfaces | 2D projection collapses depth; tokens cluster at same y | 3D unwrap or learned reading-order model |

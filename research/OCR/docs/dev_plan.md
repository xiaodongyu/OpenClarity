# Development Plan: On-Device OCR

## Goal

Capture webcam frames, apply a preprocessing pipeline, run PaddleOCR fully on-device, structure the recognised text, and speak it via TTS. Zero network dependency — the entire pipeline must work offline on Ubuntu.

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Camera capture + ROI selection | `src/capture.py` |
| 2 | Image preprocessing pipeline | `src/preprocess.py` |
| 3 | PaddleOCR engine wrapper | `src/ocr_engine.py` |
| 4 | Text structuring and filtering | `src/text_formatter.py` |
| 5 | TTS output | `src/tts.py` |
| 6 | Main pipeline loop | `src/pipeline.py` |
| 7 | Unit and integration tests | `test/` |

---

## File Structure

```
OCR/
├── docs/
│   └── dev_plan.md
├── src/
│   ├── capture.py          # Webcam capture + interactive ROI crop
│   ├── preprocess.py       # Grayscale, threshold, deskew
│   ├── ocr_engine.py       # PaddleOCR wrapper with confidence filtering
│   ├── text_formatter.py   # Left-to-right/top-to-bottom text structuring
│   ├── tts.py              # pyttsx3 TTS (offline)
│   └── pipeline.py         # Main loop
└── test/
    ├── test_preprocess.py
    ├── test_ocr_engine.py
    ├── test_text_formatter.py
    └── test_pipeline.py
```

---

## Environment Setup

**OS**: Ubuntu 22.04+  
**Python**: 3.10+

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2 espeak
python3 -m venv .venv && source .venv/bin/activate
pip install paddlepaddle paddleocr opencv-python-headless numpy pyttsx3
```

PaddleOCR downloads model weights on first run (~50 MB for the English model). Pre-download before the demo:
```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
```

---

## Phases

### Phase 1 — Camera Capture & ROI (`src/capture.py`)

**Tasks**
- Open webcam with `cv2.VideoCapture(0)`
- Expose `capture_frame() -> np.ndarray` for a single BGR frame
- Expose `select_roi(frame) -> tuple[int,int,int,int]` using `cv2.selectROI` for interactive crop selection
- Expose `crop(frame, roi) -> np.ndarray` applying the selected region
- Support `ROI_PRESET` env var (e.g. `"100,80,400,300"`) to skip interactive selection during demo

**Rationale**: Full-frame OCR on a busy scene produces garbage. ROI selection focuses the engine on the target surface (label, sign, document).

**Acceptance criteria**: `test_capture.py` verifies crop dimensions match ROI coordinates.

---

### Phase 2 — Preprocessing Pipeline (`src/preprocess.py`)

**Tasks**
- `to_grayscale(img) -> np.ndarray`
- `adaptive_threshold(img) -> np.ndarray` — `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C`
- `deskew(img) -> np.ndarray` — compute skew angle via Hough lines; rotate to correct if |angle| > 1°
- `preprocess(img) -> np.ndarray` — convenience function chaining the above steps

```python
def preprocess(img):
    gray = to_grayscale(img)
    thresh = adaptive_threshold(gray)
    return deskew(thresh)
```

**Acceptance criteria**: `test_preprocess.py` verifies output dtype is `uint8`, shape is preserved, and a synthetically skewed image is corrected to within ±1°.

---

### Phase 3 — OCR Engine (`src/ocr_engine.py`)

**Tasks**
- Initialise `PaddleOCR(use_angle_cls=True, lang='en')` once at module import (not per call)
- Expose `recognize(img: np.ndarray, conf_threshold=0.6) -> list[dict]`:
  - Returns list of `{"text": str, "confidence": float, "bbox": list}` sorted top-to-bottom, left-to-right
  - Discards tokens where `confidence < conf_threshold`
- Log recognition time to stderr

**Acceptance criteria**: `test_ocr_engine.py` runs against two known test images (stored in `test/fixtures/`) and verifies expected tokens appear in output.

---

### Phase 4 — Text Formatter (`src/text_formatter.py`)

**Tasks**
- `structure_text(tokens: list[dict]) -> str`:
  - Sort tokens by top-left `y` then `x` of bounding box (reading order)
  - Group tokens into lines by `y`-proximity (within 10 px vertical overlap)
  - Join tokens within a line with a space; join lines with a newline
- `format_for_speech(text: str) -> str`:
  - Replace newlines with pauses (`. ` or `; `)
  - Strip repeated whitespace
  - Announce field boundaries if text contains `:` patterns (e.g., `"Name: John"`)

**Acceptance criteria**: `test_text_formatter.py` provides a fixture token list and verifies the output string matches expected reading order.

---

### Phase 5 — TTS (`src/tts.py`)

**Tasks**
- Use `pyttsx3` (offline, no API cost)
- Initialise engine once; expose `speak(text: str) -> None`
- Run in a dedicated daemon thread with a priority queue — new calls interrupt the current utterance
- Configurable speech rate via `TTS_RATE` env var (default: 175 wpm)

**Acceptance criteria**: `test_tts.py` verifies the queue drains correctly and interruption does not deadlock.

---

### Phase 6 — Main Pipeline (`src/pipeline.py`)

**Tasks**
- On-demand mode (default): wait for user keypress (`Space`) to trigger one capture → preprocess → OCR → speak cycle
- Continuous mode (`--continuous`): repeat every `CAPTURE_INTERVAL_SEC` seconds (default: 3)
- Keyboard interrupt (`Ctrl+C`) exits cleanly, releasing the camera
- `--no-tts` flag prints recognised text to stdout instead of speaking (useful for testing)
- `--lang` flag passes language code to PaddleOCR (default: `en`)

**Demo scenario**: Operator presses `Space` to trigger reading of a product label, sign, or document. Three pre-selected test documents with known ground truth are stored in `test/fixtures/demo_docs/` for live accuracy validation.

**Acceptance criteria**: End-to-end run on a test image with `--no-tts` returns expected text within 2 s.

---

### Phase 7 — Testing & Demo Prep

**Tasks**
- `test_pipeline.py`: integration test using a fixture image; verify OCR output matches expected text
- Accuracy benchmark: run against 3 demo documents, compute character error rate (CER)
- Pre-download PaddleOCR model weights and commit a setup check script:
  ```bash
  python src/check_setup.py  # verifies camera, PaddleOCR model, and pyttsx3
  ```
- Write `README.md` covering setup, env vars, and demo run command

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| OCR engine | PaddleOCR | Best Python accuracy on CPU; multilingual; actively maintained |
| Trigger mode | On-demand (keypress) | Avoids continuous OCR churn; user controls when to read |
| Confidence threshold | 0.6 | Balances recall vs. noise; tunable via env var |
| TTS | `pyttsx3` | Fully offline; no API cost or latency |
| Preprocessing | Adaptive threshold + deskew | Handles varying lighting and camera angle on physical documents |

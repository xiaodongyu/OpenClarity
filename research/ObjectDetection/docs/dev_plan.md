# Development Plan: Object Detection with Spatial Audio

## Goal

Detect objects in a live webcam stream using YOLOv8n, map each detected object's horizontal position to a stereo audio pan, and emit distinct tones (earcons) so a blind user can locate objects by sound. The pipeline must run fully on-device (no network), in real time at ≥ 10 FPS on Ubuntu CPU.

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Camera capture module | `src/capture.py` |
| 2 | YOLOv8n object detector | `src/detector.py` |
| 3 | Priority filter | `src/priority_filter.py` |
| 4 | Spatial audio synthesiser | `src/spatial_audio.py` |
| 5 | Main pipeline loop | `src/pipeline.py` |
| 6 | Unit and integration tests | `test/` |

---

## File Structure

```
ObjectDetection/
├── docs/
│   └── dev_plan.md
├── src/
│   ├── capture.py           # Webcam frame capture via OpenCV
│   ├── detector.py          # YOLOv8n wrapper; returns detections with centroids
│   ├── priority_filter.py   # Limit to top-N detections by confidence
│   ├── spatial_audio.py     # Stereo pan + earcon synthesis via sounddevice
│   └── pipeline.py          # Main loop
└── test/
    ├── test_detector.py
    ├── test_priority_filter.py
    ├── test_spatial_audio.py
    └── test_pipeline.py
```

---

## Environment Setup

**OS**: Ubuntu 22.04+  
**Python**: 3.10+

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python-headless sounddevice numpy
```

YOLOv8n weights (~6 MB) auto-download on first use:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # downloads to ~/.cache/ultralytics/
```

Pre-download before the demo:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## Phases

### Phase 1 — Camera Capture (`src/capture.py`)

**Tasks**
- Open webcam with `cv2.VideoCapture(0)`
- Expose `capture_frame() -> np.ndarray` returning a single BGR frame
- Expose `get_frame_dims() -> tuple[int, int]` returning `(width, height)`
- Set capture resolution to 640×480 via `cv2.CAP_PROP_FRAME_WIDTH/HEIGHT` for speed

**Acceptance criteria**: `test_capture.py` verifies frame shape is `(480, 640, 3)` and `get_frame_dims()` returns `(640, 480)`.

---

### Phase 2 — Object Detector (`src/detector.py`)

**Tasks**
- Load `YOLOv8n` once at module init; expose `detect(frame: np.ndarray, conf=0.5) -> list[dict]`
- Each detection dict:
  ```python
  {
    "label": str,          # COCO class name, e.g. "person"
    "confidence": float,
    "bbox": (x1, y1, x2, y2),
    "centroid": (cx, cy),  # integer pixel coordinates
  }
  ```
- Filter out classes irrelevant to navigation (e.g., `"sports ball"`, `"kite"`) via `IGNORED_CLASSES` set — configurable in `detector.py`
- Log inference time per frame to stderr

**Key COCO classes for indoor/demo environments**: `person`, `chair`, `laptop`, `bottle`, `cup`, `door`, `backpack`, `cell phone`, `suitcase`, `table`

**Acceptance criteria**: `test_detector.py` loads a fixture image (`test/fixtures/indoor_scene.jpg`) and verifies at least one known object is detected with correct label and valid bounding box.

---

### Phase 3 — Priority Filter (`src/priority_filter.py`)

**Tasks**
- Expose `top_n(detections: list[dict], n=4) -> list[dict]`
  - Sort by `confidence` descending
  - Return the top `n` detections
- Limit is configurable via `MAX_DETECTIONS` env var (default: 4)

**Rationale**: Without filtering, a busy scene produces too many simultaneous tones, making audio unintelligible. Four simultaneous earcons is the practical upper limit.

**Acceptance criteria**: `test_priority_filter.py` verifies that with 10 input detections, `top_n(..., n=4)` returns exactly 4 with the highest confidence values.

---

### Phase 4 — Spatial Audio (`src/spatial_audio.py`)

**Tasks**

**Pan mapping**
```python
def centroid_to_pan(cx: int, frame_width: int) -> float:
    """Map horizontal centroid to stereo pan: -1.0 (left) to +1.0 (right)."""
    return (cx / frame_width) * 2 - 1
```

**Distance proxy**
```python
def bbox_to_proximity(bbox, frame_width, frame_height) -> float:
    """Normalised bounding box area: 0.0 (far) to 1.0 (fills frame)."""
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
```

**Earcon generation**
- Assign a distinct base frequency to each COCO class used in demo (stored in `EARCON_FREQ` dict)
- Generate a 150 ms stereo sine tone at the assigned frequency, panned by `pan`, amplitude scaled by `proximity`
- Synthesise using `numpy` + `sounddevice.play()`

```python
# Example earcon frequencies (Hz)
EARCON_FREQ = {
    "person":   440,   # A4
    "chair":    330,   # E4
    "laptop":   523,   # C5
    "bottle":   392,   # G4
    "door":     261,   # C4
}
DEFAULT_FREQ = 300  # for any class not in EARCON_FREQ
```

**Audio scheduling**
- Expose `emit(detections: list[dict], frame_dims: tuple[int,int]) -> None`
- Play all earcons for the current detection list concurrently (mix into a single stereo buffer)
- Non-blocking: use `sounddevice.play()` with `blocking=False`; new frame overrides previous

**Acceptance criteria**: `test_spatial_audio.py` verifies that a detection with `cx = 0` produces a buffer with full energy in the left channel and zero in the right, and vice versa for `cx = frame_width`.

---

### Phase 5 — Main Pipeline (`src/pipeline.py`)

**Tasks**
- Continuous loop at target 10 FPS:
  ```
  capture_frame → detect → priority_filter → spatial_audio.emit
  ```
- Keyboard interrupt (`Ctrl+C`) exits cleanly, releasing camera and audio device
- Optional visual overlay (`--visualise` flag): draw bounding boxes and labels on frame and show with `cv2.imshow` (useful for development; disabled during BLV demo)
- `--no-audio` flag: print detections to stdout instead of emitting earcons (for testing)
- `--conf FLOAT` flag: override confidence threshold (default: 0.5)

**Demo scenario**: Walk through an indoor convention space. The system continuously emits earcons indicating people, chairs, tables, and obstacles by position (left/centre/right) and proximity (loudness).

**Acceptance criteria**: End-to-end run with `--no-audio --visualise` on a fixture video file processes ≥ 10 FPS on CPU.

---

### Phase 6 — Testing & Demo Prep

**Tasks**
- `test_pipeline.py`: integration test using a fixture video (10-frame clip); verify detections are non-empty and audio `emit` is called each frame
- FPS benchmark: `python src/pipeline.py --no-audio --visualise` for 100 frames; report mean FPS
- Demo script: `demo.sh` that activates the venv, pre-checks the camera and audio device, then launches `pipeline.py`
- Write `README.md` covering setup, env vars, flag reference, and demo run instructions

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection model | YOLOv8n (nano) | Best CPU inference speed (~15 ms/frame); 6 MB weights; 80 COCO classes |
| Max detections | 4 | Cognitive limit for simultaneous audio streams |
| Audio output | `sounddevice` + `numpy` sine synthesis | No external audio files; deterministic; works on any ALSA/PulseAudio Ubuntu setup |
| Earcon design | Per-class distinct frequencies | Enables learning a fixed vocabulary of sounds per object type |
| Distance proxy | Normalised bounding box area | No depth sensor required; adequate for coarse near/far distinction |
| Visualisation | Off by default | Avoids distracting sighted operators during BLV user testing |

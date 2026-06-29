# Algorithm Reference — ObjectDetection

Detects objects in a live webcam stream and conveys their identity and position
through stereo audio tones (earcons) — no screen required.

For prompt-conditioned evaluation, the baseline task is:

```text
Input:  "where is the {object}?"
Output: one of {top,middle,bottom}-{left,middle,right}, or N/A
```

The baseline policy is intentionally simple:
- If `{object}` maps to a YOLOv8n COCO class, run detection and keep only that
  class.
- If one or more matches are found, return the location bucket of the
  highest-confidence match.
- If no match is found, return `N/A`.
- If the prompt object is outside the supported YOLO baseline vocabulary,
  return `N/A`.

```
capture_frame → detect (YOLOv8n) → priority_filter → spatial_audio.emit
```

---

## 1. Object Detection — YOLOv8n

**Model**: YOLOv8n (nano), 80 COCO classes, ~6 MB weights.

YOLOv8 is a single-pass convolutional detector. The nano variant runs at roughly
15 ms per frame on CPU (640×480 input), satisfying the ≥10 FPS target.

Each detected object is returned as:

```python
{
    "label":      str,          # COCO class name, e.g. "person"
    "confidence": float,        # 0.0–1.0
    "bbox":       (x1, y1, x2, y2),
    "centroid":   (cx, cy),     # integer pixel coordinates
}
```

A fixed `IGNORED_CLASSES` set removes classes irrelevant to navigation (e.g.
`"sports ball"`, `"kite"`, `"surfboard"`).

---

## 2. Priority Filter

With a busy scene, playing dozens of simultaneous tones is cognitively
overwhelming. The filter keeps only the top-N detections by confidence score:

```python
top_n(detections, n=4)  # configurable via MAX_DETECTIONS env var
```

Four simultaneous earcons is the practical upper limit for a listener to
parse distinct audio streams.

For prompt-conditioned evaluation, this filter is bypassed because the target
class comes from the user query rather than from generic scene ranking.

---

## 3. Spatial Audio Synthesis

Each detection is mapped to a stereo earcon that encodes both **identity**
(which object) and **position** (where it is).

### Stereo pan — horizontal position

```
pan = (cx / frame_width) × 2 − 1        # −1.0 = far left, +1.0 = far right
```

The centroid x-coordinate maps linearly to the stereo field. A person on the
left of frame produces a tone in the left ear; one on the right produces a tone
in the right ear.

### Amplitude — proximity (distance proxy)

```
proximity = bbox_area / frame_area       # 0.0 = far, 1.0 = fills frame
```

No depth sensor is required. A larger bounding box means the object is closer,
so the earcon is louder. This gives a coarse near/far cue within a single audio
channel.

### Earcon generation

Each COCO class has a fixed frequency (Hz):

| Class      | Frequency | Note |
|------------|-----------|------|
| person     | 440 Hz    | A4   |
| chair      | 330 Hz    | E4   |
| laptop     | 523 Hz    | C5   |
| bottle     | 392 Hz    | G4   |
| door       | 261 Hz    | C4   |
| cup        | 349 Hz    | F4   |
| cell phone | 587 Hz    | D5   |

Unknown classes use 300 Hz. A 150 ms sine burst is synthesised with NumPy and
mixed into a single stereo buffer. The buffer is played non-blocking via
`sounddevice.play()`, so each new frame can interrupt the previous earcon
immediately.

### Mixing and soft clipping

Multiple earcons are summed sample-by-sample. If the peak exceeds 1.0 the
entire mix is normalised to prevent clipping:

```python
if peak > 1.0:
    mix /= peak
```

---

## 4. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection model | YOLOv8n | ~15 ms CPU inference; 6 MB; 80 COCO classes |
| Max simultaneous earcons | 4 | Cognitive limit for parallel audio streams |
| Distance proxy | Normalised bbox area | No depth sensor required |
| Earcon duration | 150 ms sine burst | Short enough to update every frame; long enough to be audible |
| Visualisation | Off by default | Avoids distracting sighted operators during BLV user testing |

## 5. Evaluation Procedure

The prompt-conditioned baseline is evaluated against fixture files containing:
- image name + user prompt
- image name + target label + ground-truth location bucket

The implementation in `src/query_eval.py` evaluates each example as follows:
1. Parse the target object from the prompt.
2. Normalize the object name to a supported YOLO baseline class if possible.
3. Run YOLOv8n on the image.
4. Keep detections whose class matches the normalized target label.
5. Return the 3x3 grid location of the highest-confidence match, or `N/A`.
6. Compare the predicted location against the annotated ground truth.

This makes the benchmark closer to real usage than open-ended scene detection,
because success is defined by answering a user's specific query.

**Research Projects for OpenClarity**

This folder contains multiple research projects for OpenClarity. For each project, there is a subfolder which contains all the codes, documents and tests. 

---

## Project 1: Scene Description via VLM API

**Architecture**
- Camera frame → VLM API (GPT-4o Vision or Claude claude-sonnet-4-6 vision) → TTS output
- No on-device model; compute is entirely remote

**Stack**
```
Camera capture: OpenCV (Python)
API: OpenAI /v1/chat/completions with image_url, or Anthropic /v1/messages with image base64
TTS: ElevenLabs API, OpenAI TTS, or pyttsx3 (offline, zero API cost)
```

**Critical implementation path**
1. Capture frame at configurable interval (1–3s recommended; avoid continuous streaming)
2. Encode to base64 JPEG at 512×512 or 768×768 — larger provides marginal gain, increases latency and cost
3. Prompt engineering is the leverage point: system prompt must specify BLV context explicitly

```python
SYSTEM = """You are a visual assistant for a blind user. 
Describe what is in front of the user in 1-2 sentences.
Prioritize: people, obstacles, text, wayfinding cues.
Do not say 'I see' or 'the image shows'. Speak directly."""
```

4. Response → TTS → output; target end-to-end latency ≤3s

**Demo-critical**: Pre-cache 3–5 representative scene responses for fallback if API rate limits or connectivity fails at NFB venue.

---

## Project 2: On-Device OCR

**Architecture**
- Camera frame → on-device OCR engine → structured text → TTS
- Zero network dependency; works offline

**Stack options**

| Engine | Notes |
|--------|-------|
| PaddleOCR | Best Python option; multilingual; runs on CPU; recommended |
| EasyOCR | Better than Tesseract on complex layouts; GPU optional |
| Tesseract 5 + OpenCV preprocessing | Adequate for print; struggles with handwriting/low contrast |

**For Ubuntu demo**: PaddleOCR is the recommended choice — strong accuracy on CPU, no GPU required, handles multilingual text well. Tesseract is not demo-quality without substantial preprocessing tuning.

**Implementation path**
1. Region-of-interest crop before OCR — full-frame OCR on a busy scene produces garbage
2. Preprocessing pipeline: grayscale → adaptive threshold → deskew (if needed)
3. Confidence filtering: discard tokens below threshold (0.6 for Tesseract, built-in for Vision/ML Kit)
4. Output structuring: read text left-to-right, top-to-bottom; indicate field boundaries if reading a form

**Demo scenario**: Point camera at product label, document, menu, or sign. Have 3 pre-selected test documents with known ground truth to validate accuracy live.

---

## Project 3: Object Detection with Spatial Audio

**Architecture**
- Camera frame → object detector → bounding box centroids → positional audio synthesis
- Most complex of the three; requires tuning for demo reliability

**Model selection** (inference-only, pre-trained weights)

| Model | Weight source | Speed | Notes |
|-------|--------------|-------|-------|
| YOLOv8n | Ultralytics HuggingFace | ~15ms CPU | Best CPU inference; "n" = nano |
| YOLOv9 | Ultralytics | Slightly better mAP | Marginally slower |
| MobileNet-SSD | TensorFlow Hub | Fast on mobile | Less accurate on small objects |
| RT-DETR | Ultralytics | GPU preferred | Overkill for demo |

**For NFB demo**: YOLOv8n via Ultralytics Python SDK. One-line model load, COCO-pretrained (80 classes), runs on CPU.

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # auto-downloads ~6MB weights
results = model(frame, conf=0.5)
```

**Spatial audio mapping**
- Bounding box centroid x-coordinate → stereo pan position
- Distance proxy: use bounding box area relative to frame area (larger box = closer)
- Panning: map x ∈ [0, frame_width] → pan ∈ [-1.0, 1.0]

```python
# Pan mapping
pan = (cx / frame_width) * 2 - 1  # -1 = hard left, +1 = hard right

# Distance proxy (normalized area)
box_area = (x2-x1) * (y2-y1)
frame_area = frame_width * frame_height
proximity = box_area / frame_area  # 0 = far, 1 = fills frame
```

- Audio output: use `sounddevice` + `numpy` for real-time stereo tone synthesis, or pre-generate earcon set (distinct tones per object class)
- Priority filtering: limit to 3–5 highest-confidence detections per frame to avoid audio overload

**Demo scenario constraint**: COCO classes (person, chair, laptop, bottle, door, car, etc.) are adequate for an indoor convention floor. Do not promise detection of objects outside COCO-80 without a custom-trained model, which is out of scope.

---

## Cross-cutting implementation decisions

**Platform**: Python on Ubuntu for all three. This avoids any platform-specific build tooling and runs entirely from the command line with standard pip packages.

**Frame pipeline**: Single shared capture loop; branch per project:
```
camera frame → [scene_describer | ocr_processor | object_detector]
```
Avoids multiple camera instances competing.

**TTS**: Use a single TTS abstraction layer with priority queue — object detection audio fires continuously, OCR and scene description are interrupt-on-demand. Without priority arbitration, simultaneous audio output is unintelligible.

**Failsafe for NFB demo environment**: Convention halls have unreliable WiFi. Projects 2 and 3 are fully offline. Project 1 (VLM API) should have a local fallback — LLaVA-1.5 7B via Ollama runs on Ubuntu CPU at ~8–15s latency (faster with a discrete GPU), which is demo-acceptable.

```bash
# Install Ollama on Ubuntu
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava:7b
# Then in Python: requests.post('http://localhost:11434/api/generate', ...)
```

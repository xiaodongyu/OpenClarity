# Development Plan: Scene Description via VLM API

## Goal

Capture webcam frames at a configurable interval, send them to a VLM API for scene description, and output the result as speech. Target end-to-end latency ≤ 3 s on a reliable connection, with an offline fallback for demo environments.

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Camera capture module | `src/capture.py` |
| 2 | VLM API client (Anthropic + OpenAI) | `src/vlm_client.py` |
| 3 | TTS abstraction layer | `src/tts.py` |
| 4 | Offline fallback via Ollama/LLaVA | `src/fallback.py` |
| 5 | Main pipeline loop | `src/pipeline.py` |
| 6 | Unit and integration tests | `test/` |

---

## File Structure

```
SceneDescription/
├── docs/
│   └── dev_plan.md
├── src/
│   ├── capture.py        # Frame capture from webcam via OpenCV
│   ├── vlm_client.py     # API abstraction: Anthropic / OpenAI
│   ├── tts.py            # TTS abstraction: pyttsx3 / ElevenLabs
│   ├── fallback.py       # Local LLaVA via Ollama REST API
│   └── pipeline.py       # Main loop tying all modules together
└── test/
    ├── test_capture.py
    ├── test_vlm_client.py
    ├── test_tts.py
    └── test_pipeline.py
```

---

## Environment Setup

**OS**: Ubuntu 22.04+  
**Python**: 3.10+

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python-headless anthropic openai pyttsx3 python-dotenv requests
```

API keys stored in `.env` (never committed):
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...   # optional
```

---

## Phases

### Phase 1 — Camera Capture (`src/capture.py`)

**Tasks**
- Open webcam with `cv2.VideoCapture(0)`
- Expose `capture_frame() -> np.ndarray` returning a single BGR frame
- Expose `encode_jpeg(frame, size=(768, 768)) -> str` returning base64-encoded JPEG
- Configurable capture interval via `CAPTURE_INTERVAL_SEC` env var (default: 2.0)

**Acceptance criteria**: `test_capture.py` verifies frame shape, JPEG base64 is decodable, and interval timing is within ±100 ms.

---

### Phase 2 — VLM API Client (`src/vlm_client.py`)

**Tasks**
- Implement `describe_scene(image_b64: str) -> str` using Anthropic `claude-sonnet-4-6` by default
- Implement same interface using OpenAI `gpt-4o` as an alternative backend
- Select backend via `VLM_BACKEND` env var: `anthropic` (default) or `openai`
- Use the BLV-optimised system prompt:

```python
SYSTEM_PROMPT = """You are a visual assistant for a blind user.
Describe what is in front of the user in 1-2 sentences.
Prioritize: people, obstacles, text, wayfinding cues.
Do not say 'I see' or 'the image shows'. Speak directly."""
```

- Hard timeout of 5 s per request; raise `TimeoutError` on breach
- Log request latency to stderr

**Acceptance criteria**: `test_vlm_client.py` mocks the HTTP layer and verifies prompt construction, backend selection, and timeout behaviour.

---

### Phase 3 — TTS Abstraction (`src/tts.py`)

**Tasks**
- Implement `speak(text: str) -> None` with two backends selectable via `TTS_BACKEND` env var:
  - `pyttsx3` (default, offline): initialise once at startup; call `engine.say()` + `engine.runAndWait()`
  - `elevenlabs`: POST to ElevenLabs API, stream audio via `sounddevice`
- Priority queue: new `speak()` calls interrupt any in-progress utterance
- Thread-safe: TTS runs in a dedicated daemon thread

**Acceptance criteria**: `test_tts.py` verifies the queue drains correctly and interruption works without deadlock.

---

### Phase 4 — Offline Fallback (`src/fallback.py`)

**Tasks**
- Install Ollama on Ubuntu:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llava:7b
  ```
- Implement `describe_scene_local(image_b64: str) -> str` via `POST http://localhost:11434/api/generate`
- Expected latency: 8–15 s on CPU-only Ubuntu; acceptable for demo failsafe
- `pipeline.py` falls back automatically when the remote VLM call raises `TimeoutError` or a network error

**Acceptance criteria**: Manual test against a running Ollama instance returns a non-empty description.

---

### Phase 5 — Main Pipeline (`src/pipeline.py`)

**Tasks**
- Single loop: capture → encode → describe → speak, repeating at `CAPTURE_INTERVAL_SEC`
- Keyboard interrupt (`Ctrl+C`) exits cleanly, releasing the camera
- Pre-cache 3–5 representative responses loaded from `demo_cache.json` at startup; serve cached responses when both remote and local backends fail

```python
# demo_cache.json structure
[
  {"scene": "indoor_convention", "response": "You are in a large conference hall..."},
  ...
]
```

- CLI flags:
  - `--backend [anthropic|openai|local]` — override `VLM_BACKEND`
  - `--interval FLOAT` — override capture interval
  - `--no-tts` — print descriptions to stdout only (useful for testing)

**Acceptance criteria**: End-to-end run with `--no-tts --interval 3` completes three cycles without error.

---

### Phase 6 — Testing & Demo Prep

**Tasks**
- `test_pipeline.py`: integration test using mocked VLM client and TTS; verify three full cycles complete
- Latency benchmark script: `python src/pipeline.py --no-tts --interval 0` for 10 frames, report mean/p95
- Populate `demo_cache.json` with pre-recorded responses for 5 representative scenes
- Write `README.md` in project root covering setup, env vars, and demo run command

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary VLM | Anthropic `claude-sonnet-4-6` | Strong vision capability; easy image input via base64 |
| Image resolution | 768×768 | Balance between detail and API cost/latency |
| TTS default | `pyttsx3` | Zero API cost, works offline, no added latency |
| Offline fallback | LLaVA-1.5 7B via Ollama | Runs on Ubuntu CPU; single-command install |
| Capture interval | 2 s default | Avoids redundant API calls on slow-moving scenes |

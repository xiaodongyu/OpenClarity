# SceneDescription — Scene Description via VLM API

Captures webcam frames at a configurable interval, sends them to a VLM API for scene description, and speaks the result. Target end-to-end latency ≤ 3 s on a reliable connection, with an offline fallback for demo environments.

## Requirements

- Ubuntu 22.04+
- Python 3.10+
- Webcam (device `/dev/video0`)

## Setup

```bash
sudo apt-get install -y python3-dev python3-venv libportaudio2
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python-headless anthropic openai pyttsx3 python-dotenv requests
```

Create a `.env` file (never committed):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...   # optional
```

### Offline fallback (Ollama/LLaVA)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava:7b
```

## Usage

### Default (Anthropic, pyttsx3 TTS)

```bash
python src/pipeline.py
```

### Override backend

```bash
python src/pipeline.py --backend openai
python src/pipeline.py --backend local   # Ollama only
```

### No-TTS mode (prints to stdout)

```bash
python src/pipeline.py --no-tts
```

### Custom capture interval

```bash
python src/pipeline.py --interval 5
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key (optional) |
| `VLM_BACKEND` | `anthropic` | `anthropic` or `openai` |
| `TTS_BACKEND` | `pyttsx3` | `pyttsx3` or `elevenlabs` |
| `CAPTURE_INTERVAL_SEC` | `2.0` | Seconds between captures |

## Fallback chain

1. Remote VLM (Anthropic or OpenAI)
2. Local Ollama/LLaVA (`http://localhost:11434`)
3. Pre-cached responses from `demo_cache.json`

## Demo

Five pre-recorded scene responses are stored in `demo_cache.json` (convention hall, street, office corridor, cafe, parking lot) and are served when all backends fail.

Run a latency benchmark (10 frames, no speech):

```bash
python src/pipeline.py --no-tts --interval 0
```

## Tests

```bash
pytest test/ -v
```

All 26 tests run offline — no camera, API keys, or Ollama instance required.

## Evaluation

`test/eval_outdoor.py` runs the full remote-VLM pipeline on static test images and scores each generated description against a one-sentence ground truth using an LLM judge (semantic similarity, 0.0–1.0). Results are written as `eval_<YYYYMMDD>_<NNN>.json` and `.html` to the fixture folder.

### Fixture: outdoor (3 images)

| Image | Scene | Score |
|---|---|---|
| `image_q1.png` | Quiet residential street with trees | 0.85 |
| `image_b1.png` | Pedestrian crosswalk on a busy road | 0.85 |
| `image_f1.png` | Front door of a house with green plants | 0.75 |

**Result: PASS — 3/3 (mean score 0.82, threshold ≥ 0.70)**  
Tested at commit `09d43a5` · 2026-06-29

### Run the eval

```bash
# from research/SceneDescription/
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
.venv/bin/python test/eval_outdoor.py
```

Ground truth is read from `test/fixtures/outdoor/scene_description_groundtruth.txt` (format: `filename: one sentence`). Add more fixture subfolders with the same file to extend coverage.

### Scoring

The LLM judge (claude-sonnet-4-6) compares the generated description to the ground truth and returns a score:

| Score | Meaning |
|---|---|
| ≥ 0.70 | PASS — essential information conveyed |
| 0.40–0.69 | Partial match — key elements missing |
| < 0.40 | Fail — wrong or empty scene |

> **Note:** The eval bypasses the pipeline's 5-second remote timeout so the full VLM output quality is visible. In live use, descriptions taking > 5s fall back to Ollama/LLaVA.

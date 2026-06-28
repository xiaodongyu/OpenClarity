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

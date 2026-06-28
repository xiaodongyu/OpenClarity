# Algorithm Reference — SceneDescription

Periodically captures a camera frame, sends it to a Vision Language Model
(VLM), and speaks the resulting natural-language description.

```
capture_frame → encode JPEG → describe_scene (VLM) → TTS
                                     ↓ on timeout/error
                              describe_scene_local (Ollama/LLaVA)
                                     ↓ on error
                              demo_cache.json (round-robin)
```

---

## 1. Remote VLM Client

`src/vlm_client.py: describe_scene` dispatches to one of two backends selected
by the `VLM_BACKEND` environment variable:

| Backend | Model | API |
|---------|-------|-----|
| `anthropic` (default) | `claude-sonnet-4-6` | Anthropic Messages API |
| `openai` | `gpt-4o` | OpenAI Chat Completions API |

The frame is base64-encoded as a JPEG and sent as an image content block. The
system prompt instructs the model to describe the scene in 1–2 sentences,
prioritising people, obstacles, text, and wayfinding cues, without filler
phrases like "I see" or "the image shows".

A hard **5-second timeout** is applied to both backends. Each backend's native
timeout exception is caught and re-raised as Python's built-in `TimeoutError`
so the fallback chain handles a single exception type regardless of which
backend is active.

---

## 2. Three-Tier Fallback Chain

The pipeline degrades gracefully when the network or the remote API is
unavailable.

### Tier 1 — Remote VLM (default)

Calls `describe_scene()` with the configured backend. On `TimeoutError` or any
network/OS error, falls through to Tier 2.

### Tier 2 — Local Ollama / LLaVA 7B

`src/fallback.py: describe_scene_local` posts the base64 image to a locally
running Ollama instance at `localhost:11434` serving `llava:7b`. This path
works without internet connectivity. Timeout is 60 seconds (inference on CPU
can be slow). On failure, falls through to Tier 3.

### Tier 3 — Demo cache

`demo_cache.json` holds a list of pre-recorded scene descriptions. The pipeline
selects an entry by `int(time.time()) % len(cache)`, cycling through the list
by wall-clock second. This guarantees the system always produces audio output
during demos regardless of connectivity or hardware availability.

---

## 3. Text-to-Speech

Descriptions are enqueued into a priority queue consumed by a daemon TTS
thread. If a new description arrives before the current one finishes speaking,
the queue is drained so only the latest description is spoken. This prevents
stale descriptions from accumulating when the scene changes rapidly.

---

## 4. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default VLM backend | Anthropic (`claude-sonnet-4-6`) | Strong multimodal understanding; falls back to OpenAI via env var |
| VLM timeout | 5 s | Hard cutoff; network latency must not block TTS |
| Local fallback model | LLaVA 7B via Ollama | Runs fully on-device; no API key required |
| Fallback chain depth | 3 tiers | Guarantees audio output in any connectivity scenario |
| TTS interruption | Daemon thread + priority queue; drain on wakeup | Avoids stale descriptions when scene changes |
| Capture interval | 2 s (default, configurable) | Balances freshness with API cost and TTS duration |

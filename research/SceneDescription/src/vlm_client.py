import os
import sys
import time

SYSTEM_PROMPT = """You are a visual assistant for a blind user.
Describe what is in front of the user in 1-2 sentences.
Prioritize: people, obstacles, text, wayfinding cues.
Do not say 'I see' or 'the image shows'. Speak directly."""

_TIMEOUT = 5.0
_MODEL_ANTHROPIC = "claude-sonnet-4-6"
_MODEL_OPENAI = "gpt-4o"


def _describe_anthropic(image_b64: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    t0 = time.perf_counter()
    try:
        message = client.messages.create(
            model=_MODEL_ANTHROPIC,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "Describe this scene."},
                    ],
                }
            ],
            timeout=_TIMEOUT,
        )
    except Exception as exc:
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            raise TimeoutError("Anthropic request timed out") from exc
        raise
    elapsed = time.perf_counter() - t0
    print(f"[vlm_client] anthropic latency: {elapsed:.3f}s", file=sys.stderr)
    return message.content[0].text


def _describe_openai(image_b64: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=_TIMEOUT)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=_MODEL_OPENAI,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": "Describe this scene."},
                    ],
                },
            ],
        )
    except openai.APITimeoutError as exc:
        raise TimeoutError("OpenAI request timed out") from exc
    elapsed = time.perf_counter() - t0
    print(f"[vlm_client] openai latency: {elapsed:.3f}s", file=sys.stderr)
    return response.choices[0].message.content


def describe_scene(image_b64: str) -> str:
    backend = os.environ.get("VLM_BACKEND", "anthropic").lower()
    if backend == "openai":
        return _describe_openai(image_b64)
    return _describe_anthropic(image_b64)

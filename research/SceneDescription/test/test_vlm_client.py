"""
Tests for vlm_client.py.

Both anthropic and openai SDKs are mocked per-test via patch.dict so
no real API calls or installed packages are required.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

import src.vlm_client as vlm

IMAGE_B64 = "aGVsbG8="  # "hello" in base64 — valid placeholder


def _make_anthropic_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_openai_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


# ------------------------------------------------------------------
# Anthropic backend tests
# ------------------------------------------------------------------

def test_anthropic_backend_selected_by_default(monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response("A busy street.")
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = vlm.describe_scene(IMAGE_B64)

    assert result == "A busy street."
    mock_client.messages.create.assert_called_once()


def test_anthropic_prompt_construction(monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response("desc")
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        vlm.describe_scene(IMAGE_B64)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == vlm.SYSTEM_PROMPT
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    content = call_kwargs["messages"][0]["content"]
    image_block = next(b for b in content if b.get("type") == "image")
    assert image_block["source"]["data"] == IMAGE_B64
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_anthropic_timeout_raises(monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("request timed out after 5s")
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        with pytest.raises(TimeoutError):
            vlm.describe_scene(IMAGE_B64)


# ------------------------------------------------------------------
# OpenAI backend tests
# ------------------------------------------------------------------

def test_openai_backend_selected_via_env(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "openai")
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("A park.")
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = vlm.describe_scene(IMAGE_B64)

    assert result == "A park."
    mock_client.chat.completions.create.assert_called_once()


def test_openai_prompt_construction(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "openai")
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("desc")
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        vlm.describe_scene(IMAGE_B64)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    system_msg = call_kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == vlm.SYSTEM_PROMPT


def test_openai_timeout_raises(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "openai")
    mock_openai = MagicMock()
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    mock_openai.APITimeoutError = APITimeoutError
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError("timeout")
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        with pytest.raises(TimeoutError):
            vlm.describe_scene(IMAGE_B64)

"""Unit tests for `OllamaClient`, using mocked HTTP calls (no real Ollama server needed)."""
from unittest.mock import Mock, patch

import requests

from ollama_client import OllamaClient


def _make_client(enabled: bool = True) -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434/api/generate",
        model="llama3.2",
        enabled=enabled,
        timeout=5,
    )


def test_is_available_returns_false_when_disabled():
    client = _make_client(enabled=False)
    assert client.is_available() is False


@patch("ollama_client.requests.get")
def test_is_available_returns_true_when_server_responds(mock_get):
    mock_get.return_value = Mock(status_code=200)
    client = _make_client()

    assert client.is_available() is True
    mock_get.assert_called_once_with("http://localhost:11434", timeout=2)


@patch("ollama_client.requests.get", side_effect=requests.exceptions.ConnectionError)
def test_is_available_returns_false_on_connection_error(_mock_get):
    client = _make_client()
    assert client.is_available() is False


def test_generate_returns_none_when_disabled():
    client = _make_client(enabled=False)
    assert client.generate("hello") is None


@patch("ollama_client.requests.post")
def test_generate_returns_trimmed_response_text(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {"response": "  Sure thing!  "}
    mock_post.return_value = mock_response

    client = _make_client()
    assert client.generate("How do I prepare?") == "Sure thing!"


@patch("ollama_client.requests.post", side_effect=requests.exceptions.Timeout)
def test_generate_returns_none_on_timeout(_mock_post):
    client = _make_client()
    assert client.generate("How do I prepare?") is None

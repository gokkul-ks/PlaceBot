"""
Minimal client for a locally running Ollama instance (https://ollama.ai).

Ollama lets you run open LLMs (Llama 3.2, Mistral, Phi-3, ...) on your own
machine. PlaceBot calls it to produce more natural, conversational
responses when the dataset's confidence in a match is low or medium.

Every method here fails *softly*: if Ollama isn't running, times out, or
returns something unexpected, methods return ``None``/``False`` instead of
raising, so the chatbot can fall back to dataset-only responses.
"""
from typing import Optional

import requests


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str, enabled: bool, timeout: int = 30) -> None:
        self.base_url = base_url
        self.model = model
        self.enabled = enabled
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return ``True`` if an Ollama server is reachable, ``False`` otherwise."""
        if not self.enabled:
            return False

        # Ollama serves a small landing page at its root URL (e.g.
        # "http://localhost:11434/") - strip the "/api/..." suffix from the
        # generate endpoint to reach it.
        root_url = self.base_url.split("/api/")[0]
        try:
            response = requests.get(root_url, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, context: str = "") -> Optional[str]:
        """
        Ask Ollama to generate a response to ``prompt``.

        If ``context`` is provided, it's included so the model can ground its
        answer in PlaceBot's dataset. Returns ``None`` if Ollama is disabled,
        unreachable, times out, or returns an unexpected response - callers
        should treat ``None`` as "use a fallback response instead".
        """
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "prompt": self._build_prompt(prompt, context),
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 150,
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            text = response.json().get("response", "")
            return text.strip() or None
        except requests.exceptions.Timeout:
            print("⚠️  Ollama request timed out - using fallback response")
        except requests.exceptions.ConnectionError:
            print("⚠️  Could not connect to Ollama - using fallback response")
        except requests.exceptions.RequestException as exc:
            print(f"⚠️  Ollama request failed: {exc}")
        except ValueError as exc:  # response.json() found invalid JSON
            print(f"⚠️  Ollama returned an unexpected response: {exc}")

        return None

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        """Build the prompt sent to Ollama, optionally grounded in ``context``."""
        intro = (
            "You are PlaceBot, a helpful placement preparation assistant "
            "for college students.\n\n"
        )
        if context:
            return (
                f"{intro}"
                f"Context from knowledge base: {context}\n\n"
                f"User question: {question}\n\n"
                "Provide a helpful, concise and friendly response based on "
                "the context. If the context is relevant, use it. Keep your "
                "response under 100 words and be encouraging."
            )
        return (
            f"{intro}"
            f"User question: {question}\n\n"
            "Provide helpful placement guidance. Keep your response under "
            "100 words and be friendly and encouraging."
        )

"""
Application configuration for PlaceBot.

Every setting below can be overridden via an environment variable, or by
creating a ``.env`` file in the project root (see ``.env.example`` for the
full list of options and their defaults). Values are read once, when this
module is first imported.
"""
import os

from dotenv import load_dotenv

# Load variables from a local `.env` file (if one exists) into the process
# environment before reading any settings below. Safe to call even if no
# `.env` file is present.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean setting (accepts true/false/1/0/yes/no, case-insensitive)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """Read an integer setting, falling back to ``default`` on invalid input."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠️  Invalid value for {name}={value!r}, using default {default}")
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float setting, falling back to ``default`` on invalid input."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"⚠️  Invalid value for {name}={value!r}, using default {default}")
        return default


class Settings:
    """Central, read-only configuration used throughout PlaceBot."""

    # --- Ollama (optional local LLM) -------------------------------------
    # See https://ollama.ai - set OLLAMA_ENABLED=false to run PlaceBot in
    # dataset-only mode without needing Ollama installed at all.
    OLLAMA_ENABLED: bool = _get_bool("OLLAMA_ENABLED", True)
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: int = _get_int("OLLAMA_TIMEOUT", 30)

    # --- Hybrid response confidence thresholds (0.0 - 1.0) ----------------
    #   score >= HIGH_CONFIDENCE          -> answer directly from the dataset
    #   MEDIUM_CONFIDENCE <= score < HIGH -> ask Ollama, using the dataset
    #                                         answer as context
    #   score <  MEDIUM_CONFIDENCE        -> ask Ollama with no context, or
    #                                         fall back to a generic message
    HIGH_CONFIDENCE: float = _get_float("HIGH_CONFIDENCE", 0.5)
    MEDIUM_CONFIDENCE: float = _get_float("MEDIUM_CONFIDENCE", 0.3)

    # --- Dataset & embedding model ----------------------------------------
    CSV_FILE: str = os.getenv("CSV_FILE", "data.csv")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Server / browser ---------------------------------------------------
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = _get_int("PORT", 8000)
    # Automatically open the chat UI in the default browser on startup.
    # Set to "false" for server/production/CI environments.
    AUTO_OPEN_BROWSER: bool = _get_bool("AUTO_OPEN_BROWSER", True)


settings = Settings()

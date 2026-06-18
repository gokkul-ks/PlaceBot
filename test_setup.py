#!/usr/bin/env python3
"""
PlaceBot - Quick Setup Verification Script

Run this before starting the server to check that your environment is
ready:

    python test_setup.py

Note: this is a standalone diagnostic CLI, not a pytest test module - the
project's pytest suite lives in `tests/` (see `pytest.ini`).
"""
import os
import sys


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _get_settings():
    """Best-effort import of the app's settings (may not be installed yet)."""
    try:
        from config import settings
        return settings
    except ImportError:
        return None


def check_python_version() -> bool:
    """Check that the running Python version is supported."""
    print_header("1. Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    # Compare as a tuple, not major/minor independently - e.g. comparing
    # major and minor separately would incorrectly reject a future
    # Python 4.0 (4 >= 3 but 0 >= 10 is False).
    if (version.major, version.minor) >= (3, 10):
        print("✅ Python version is compatible")
        return True

    print("❌ Python 3.10+ is required")
    return False


def check_dependencies() -> bool:
    """Check that all required packages are installed."""
    print_header("2. Checking Dependencies")

    required = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "dotenv": "python-dotenv",
        "sentence_transformers": "Sentence Transformers",
        "sklearn": "Scikit-learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "requests": "Requests",
    }

    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def check_dataset() -> bool:
    """Check that the Q&A dataset CSV exists and is well-formed."""
    print_header("3. Checking Dataset")

    settings = _get_settings()
    csv_file = settings.CSV_FILE if settings else "data.csv"

    if not os.path.exists(csv_file):
        print(f"❌ '{csv_file}' not found")
        print(f"Please make sure '{csv_file}' is in the project root.")
        return False

    print(f"✅ '{csv_file}' found")

    try:
        import pandas as pd

        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        print(f"✅ Dataset loaded: {len(df)} Q&A pairs")

        if "question" not in df.columns or "answer" not in df.columns:
            print("❌ CSV must have 'question' and 'answer' columns")
            return False

        if df[["question", "answer"]].isnull().any().any():
            print("❌ CSV contains empty 'question' or 'answer' cells")
            return False

        print("✅ Dataset structure looks correct")
        return True

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def check_static_files() -> bool:
    """Check that the frontend's static files exist."""
    print_header("4. Checking Static Files")

    if not os.path.exists("static"):
        print("⚠️  'static/' folder not found - creating it...")
        os.makedirs("static", exist_ok=True)
        print("⚠️  Please add 'index.html' to the 'static/' folder")
        return True

    print("✅ 'static/' folder exists")

    if os.path.exists("static/index.html"):
        print("✅ 'index.html' found")
    else:
        print("⚠️  'index.html' not found in 'static/'")
        print("   The API will work, but the chat UI won't display")

    return True


def check_ollama() -> bool:
    """Check whether a local Ollama server is reachable (optional)."""
    print_header("5. Checking Ollama (Optional)")

    try:
        import requests
    except ImportError:
        print("⚠️  'requests' is not installed - skipping Ollama check")
        return False

    settings = _get_settings()
    base_url = settings.OLLAMA_URL if settings else "http://localhost:11434/api/generate"
    root_url = base_url.split("/api/")[0]

    try:
        response = requests.get(root_url, timeout=2)
    except requests.exceptions.RequestException:
        print("⚠️  Ollama not running (this is optional)")
        print("   PlaceBot will work in dataset-only mode")
        print("   To enable AI responses: install Ollama from https://ollama.ai")
        return False

    if response.status_code != 200:
        print(f"⚠️  Ollama responded with status {response.status_code}")
        return False

    print(f"✅ Ollama is running at {root_url}")

    try:
        models_response = requests.get(f"{root_url}/api/tags", timeout=2)
        models = models_response.json().get("models", []) if models_response.status_code == 200 else []
        if models:
            print("✅ Available models:")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print("⚠️  No models found. Run: ollama pull llama3.2")
    except requests.exceptions.RequestException:
        pass  # Listing models is a nice-to-have; Ollama itself is reachable.

    return True


def test_model_loading() -> bool:
    """Check that the sentence-embedding model can be downloaded/loaded."""
    print_header("6. Testing Semantic Model")

    settings = _get_settings()
    model_name = settings.EMBEDDING_MODEL if settings else "all-MiniLM-L6-v2"

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading '{model_name}' (this may take a minute the first time)...")
        model = SentenceTransformer(model_name)
        print("✅ Semantic model loaded successfully")

        test_embedding = model.encode(["test question"])
        print(f"✅ Model encoding works (dimension: {len(test_embedding[0])})")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def main() -> None:
    print("\n" + "🤖" * 30)
    print("PlaceBot - Setup Verification")
    print("🤖" * 30)

    results = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Dataset", check_dataset()),
        ("Static Files", check_static_files()),
        ("Ollama", check_ollama()),
        ("Semantic Model", test_model_loading()),
    ]

    print_header("SUMMARY")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"{name:20s} {'✅ PASS' if ok else '❌ FAIL'}")

    print(f"\n{passed}/{total} checks passed")

    # Python version, dependencies, and dataset are required; Ollama and
    # the semantic model check are best-effort / informational.
    critical_ok = all(ok for _, ok in results[:3])
    if critical_ok:
        print("\n✅ PlaceBot is ready to run!")
        print("\nStart with:")
        print("  python main.py")
        print("  (or: uvicorn main:app --reload)")
        print("\nOr with a custom host/port:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ Please fix the issues above before running PlaceBot")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

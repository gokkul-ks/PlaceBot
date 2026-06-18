"""
PlaceBot - AI-powered placement preparation assistant (FastAPI backend).

Combines a CSV-based Q&A dataset (matched via sentence-transformer
embeddings) with an optional local Ollama LLM for more natural responses
when the dataset doesn't have a confident match.

Run with:
    uvicorn main:app --reload

or simply:
    python main.py
"""
import threading
import traceback
import webbrowser
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from chatbot import PlacementChatbot
from config import settings
from ollama_client import OllamaClient
from schemas import ChatRequest, ChatResponse

# Populated during the `lifespan` startup hook below, and read by the route
# handlers further down this file.
chatbot: Optional[PlacementChatbot] = None


def _open_browser() -> None:
    """Open the chat UI in the user's default browser shortly after startup."""
    # "0.0.0.0" means "listen on all interfaces" - it isn't a valid URL to
    # open in a browser, so point the browser at localhost instead.
    host = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
    webbrowser.open(f"http://{host}:{settings.PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the dataset and embedding model once when the server starts.

    Doing this here (instead of at module import time) keeps heavy ML setup
    out of the way for tooling that simply imports `main` (e.g. test
    runners), and gives FastAPI a clear place to report startup failures -
    if `PlacementChatbot()` raises, the server won't start serving requests
    with a half-initialized chatbot.
    """
    global chatbot

    print("=" * 60)
    print("🤖 PlaceBot - Intelligent Placement Assistant")
    print("=" * 60)

    ollama = OllamaClient(
        base_url=settings.OLLAMA_URL,
        model=settings.OLLAMA_MODEL,
        enabled=settings.OLLAMA_ENABLED,
        timeout=settings.OLLAMA_TIMEOUT,
    )
    chatbot = PlacementChatbot(
        csv_path=settings.CSV_FILE,
        embedding_model_name=settings.EMBEDDING_MODEL,
        ollama=ollama,
    )

    print("=" * 60)
    print("🚀 PlaceBot is ready!")
    print(f"💡 Chat UI : http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API docs: http://{settings.HOST}:{settings.PORT}/docs")
    print("=" * 60)

    if settings.AUTO_OPEN_BROWSER:
        threading.Timer(1.5, _open_browser).start()

    yield  # The application serves requests while suspended here.

    print("👋 PlaceBot shutting down")


app = FastAPI(
    title="PlaceBot - AI Placement Assistant",
    description="Hybrid CSV + Ollama chatbot that answers placement-preparation questions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend to call the API. `allow_credentials=False` is required
# (and is the only valid choice) when `allow_origins` is the wildcard "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    """Serve the chat UI."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>PlaceBot backend is running!</h1>"
            "<p>Frontend not found - create <code>static/index.html</code>.</p>"
            '<p>Browse the API at <a href="/docs">/docs</a>.</p>'
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a placement-related question using the hybrid dataset/Ollama pipeline."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="PlaceBot is still starting up - please retry shortly.")

    try:
        return chatbot.get_response(request.message)
    except Exception:
        # Last-resort safety net: never let an unexpected error surface as a
        # raw 500 - log it for debugging and return a friendly message.
        traceback.print_exc()
        return ChatResponse(
            reply="Sorry, I ran into an error while processing that. Please try again.",
            confidence=0.0,
            source="error",
        )


@app.get("/health")
def health() -> dict:
    """Report whether the chatbot and Ollama integration are ready."""
    if chatbot is None:
        return {"status": "starting"}

    ollama_ready = chatbot.ollama.enabled and chatbot.ollama_available
    return {
        "status": "healthy",
        "dataset_loaded": len(chatbot.questions) > 0,
        "total_questions": len(chatbot.questions),
        "ollama_enabled": chatbot.ollama.enabled,
        "ollama_available": chatbot.ollama_available if chatbot.ollama.enabled else False,
        "model": chatbot.ollama.model if chatbot.ollama.enabled else None,
        "mode": "hybrid" if ollama_ready else "dataset-only",
    }


@app.get("/stats")
def stats() -> dict:
    """Return basic dataset and configuration statistics."""
    if chatbot is None:
        return {"status": "starting"}

    return {
        "total_questions": len(chatbot.questions),
        "dataset_file": settings.CSV_FILE,
        "semantic_model": settings.EMBEDDING_MODEL,
        "ollama_model": chatbot.ollama.model if chatbot.ollama.enabled else None,
        "thresholds": {
            "high": settings.HIGH_CONFIDENCE,
            "medium": settings.MEDIUM_CONFIDENCE,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)

"""Pydantic request/response models for the PlaceBot API."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """A chat message sent by the user to ``POST /chat``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's question or message.",
        examples=["How do I prepare for placements?"],
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        """Reject messages that are empty or contain only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or contain only whitespace.")
        return stripped


class ChatResponse(BaseModel):
    """PlaceBot's reply to a chat message."""

    reply: str = Field(..., description="The chatbot's response text.")
    confidence: float = Field(
        ..., description="Similarity score of the best dataset match (0.0 - 1.0)."
    )
    source: str = Field(
        ...,
        description=(
            "Where the reply came from: 'greeting', 'thanks', 'bye', "
            "'dataset', 'dataset-fallback', 'ollama-hybrid', 'ollama-pure', "
            "'fallback', or 'error'."
        ),
    )
    matched_question: Optional[str] = Field(
        default=None,
        description="The dataset question that best matched the user's input, if any.",
    )
    context_used: Optional[str] = Field(
        default=None,
        description="The dataset question used as context for an Ollama-generated reply, if any.",
    )

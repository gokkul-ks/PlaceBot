"""Unit tests for the API request/response models in `schemas`."""
import pytest
from pydantic import ValidationError

from schemas import ChatRequest


def test_chat_request_accepts_normal_message():
    request = ChatRequest(message="How do I prepare for placements?")
    assert request.message == "How do I prepare for placements?"


def test_chat_request_strips_surrounding_whitespace():
    request = ChatRequest(message="  hello  ")
    assert request.message == "hello"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_rejects_overly_long_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="a" * 501)

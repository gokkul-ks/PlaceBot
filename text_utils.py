"""
Lightweight, dependency-free text-processing helpers used by PlaceBot.

These functions handle the *rule-based* parts of the chatbot - greeting
detection, keyword shortcuts, and basic text normalization. They are kept
separate from `chatbot.py` so they can be unit-tested quickly without
pulling in the much heavier sentence-transformers / scikit-learn stack.
"""
import string
from typing import Dict, Optional

# --------------------------------------------------------------------------
# Rule-based replies for common conversational openers/closers
# --------------------------------------------------------------------------
GREETINGS = {"hi", "hello", "hey", "good morning", "good evening"}
THANKS_WORDS = ("thanks", "thank you", "thankyou", "appreciate")
BYE_WORDS = ("bye", "goodbye", "see you", "exit")

GREETING_REPLY = (
    "Hello! 👋 I'm PlaceBot, your AI-powered placement assistant. I can "
    "help you with placements, resume building, CGPA, coding interviews, "
    "and more. What would you like to know?"
)
THANKS_REPLY = (
    "You're welcome! 😊 Best of luck for your placements! Feel free to "
    "ask if you need more help."
)
BYE_REPLY = (
    "Goodbye! 👋 Wishing you great success in your placements and career. "
    "Keep practicing and stay confident! 🚀"
)

# --------------------------------------------------------------------------
# Keyword shortcuts
# --------------------------------------------------------------------------
# Maps a single keyword to a "canonical" question phrasing that's likely to
# have a strong match in the dataset (e.g. a user typing just "cgpa" gets
# routed to the full "minimum cgpa required for placements" question).
#
# Expansion is only applied to SHORT queries (see
# KEYWORD_EXPANSION_MAX_WORDS below) using WHOLE-WORD matches. This keeps it
# acting purely as a shortcut for short/ambiguous queries (e.g. "cgpa",
# "resume", "what is hr") without overriding semantic search on longer,
# already-specific questions - e.g. "difference between service and
# product company" should NOT be hijacked just because it contains the
# word "company".
KEYWORD_MAP: Dict[str, str] = {
    "companies": "what companies visit campus",
    "company": "what companies visit campus",
    "cgpa": "minimum cgpa required for placements",
    "resume": "how to build resume",
    "dsa": "important dsa topics for placements",
    "internship": "importance of internship",
    "hr": "common hr interview questions",
    "aptitude": "how to prepare for aptitude",
    "projects": "important projects for placements",
    "salary": "highest paying companies in placements",
    "package": "highest paying companies in placements",
}

# Keyword expansion only applies when the preprocessed query has at most
# this many words. Longer questions are usually specific enough for
# semantic search to handle well on their own.
KEYWORD_EXPANSION_MAX_WORDS = 3


def preprocess(text: str) -> str:
    """Lowercase, strip surrounding whitespace, and remove punctuation."""
    text = text.lower().strip()
    return text.translate(str.maketrans("", "", string.punctuation))


def rule_based_reply(processed_input: str) -> Optional[dict]:
    """
    Return a canned reply for greetings/thanks/goodbyes, or ``None`` if the
    input doesn't match any of those categories.

    ``processed_input`` should already be the output of :func:`preprocess`.
    """
    if processed_input in GREETINGS:
        return {"reply": GREETING_REPLY, "confidence": 1.0, "source": "greeting"}

    if any(word in processed_input for word in THANKS_WORDS):
        return {"reply": THANKS_REPLY, "confidence": 1.0, "source": "thanks"}

    if any(word in processed_input for word in BYE_WORDS):
        return {"reply": BYE_REPLY, "confidence": 1.0, "source": "bye"}

    return None


def expand_keywords(processed_input: str) -> str:
    """
    Expand a short query containing a known keyword into a fuller question
    that's more likely to match the dataset well.

    ``processed_input`` should already be the output of :func:`preprocess`.
    Returns the input unchanged if it's too long or contains no recognized
    keyword.
    """
    words = processed_input.split()
    if len(words) > KEYWORD_EXPANSION_MAX_WORDS:
        return processed_input

    for keyword, expansion in KEYWORD_MAP.items():
        if keyword in words:
            return expansion

    return processed_input

"""
Core chatbot engine for PlaceBot.

:class:`PlacementChatbot` loads the Q&A dataset and the sentence-embedding
model once at startup, then answers questions using a three-tier hybrid
strategy:

1. **Rule-based replies** for greetings, thanks, and goodbyes.
2. **Semantic search** over the dataset using sentence embeddings + cosine
   similarity.
3. **Keyword-shortcut fallback**: if step 2 doesn't reach high confidence,
   short queries containing a known keyword (e.g. "cgpa") are expanded into
   a fuller question and re-searched - the better of the two results wins.
4. The final similarity score is routed based on confidence:

   - *High confidence*   -> answer directly from the dataset.
   - *Medium confidence* -> ask Ollama, using the matched dataset answer as
     context (falls back to the dataset answer if Ollama is unavailable).
   - *Low confidence*    -> ask Ollama with no context, or return a generic
     "I'm not sure" message with suggested topics.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from ollama_client import OllamaClient
from text_utils import expand_keywords, preprocess, rule_based_reply

FALLBACK_REPLY = (
    "I'm not completely sure about that specific question. However, I can "
    "help you with:\n\n"
    "• Placement preparation strategies\n"
    "• CGPA requirements\n"
    "• Resume building tips\n"
    "• DSA and coding interview prep\n"
    "• Company information\n"
    "• Interview tips and HR rounds\n\n"
    "Could you rephrase your question or ask about one of these topics?"
)


class PlacementChatbot:
    """Loads the dataset + embedding model and answers placement-related questions."""

    def __init__(self, csv_path: str, embedding_model_name: str, ollama: OllamaClient) -> None:
        self.ollama = ollama

        print(f"📊 Loading dataset from '{csv_path}'...")
        # `utf-8-sig` transparently strips a UTF-8 BOM if the CSV was saved
        # from Excel, while behaving identically to plain `utf-8` otherwise.
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self._validate_dataset()

        self.questions = [preprocess(q) for q in self.df["question"]]
        self.answers = self.df["answer"].tolist()
        print(f"✅ Loaded {len(self.questions)} Q&A pairs")

        print(f"🧠 Loading semantic model '{embedding_model_name}' (this can take a moment)...")
        self.model = SentenceTransformer(embedding_model_name)
        print("✅ Semantic model loaded")

        print("🔄 Creating question embeddings...")
        self.question_embeddings = self.model.encode(self.questions)
        print("✅ Embeddings ready")

        self.ollama_available = self.ollama.is_available()
        self._log_mode()

    def _validate_dataset(self) -> None:
        """Validate that the loaded CSV has the columns/rows the chatbot needs."""
        required_columns = {"question", "answer"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Dataset CSV is missing required column(s): {', '.join(sorted(missing))}. "
                "Expected columns: 'question', 'answer'."
            )

        if self.df.empty:
            raise ValueError("Dataset CSV contains no rows.")

        if self.df[["question", "answer"]].isnull().any().any():
            raise ValueError(
                "Dataset CSV contains empty 'question' or 'answer' cells. "
                "Please fill in or remove incomplete rows."
            )

    def _log_mode(self) -> None:
        """Print the chatbot's response mode (hybrid vs. dataset-only) at startup."""
        if self.ollama.enabled and self.ollama_available:
            print(f"✅ Ollama is running (model: {self.ollama.model})")
            print("🎯 Mode: HYBRID (dataset + Ollama)")
        elif self.ollama.enabled:
            print("⚠️  Ollama not detected - running in dataset-only mode")
            print("💡 To enable AI responses, install Ollama from https://ollama.ai")
            print("🎯 Mode: DATASET ONLY")
        else:
            print("ℹ️  Ollama disabled in configuration")
            print("🎯 Mode: DATASET ONLY")

    # ----------------------------------------------------------------
    # Response generation
    # ----------------------------------------------------------------
    def get_response(self, user_input: str) -> dict:
        """Return a reply dict (matching :class:`schemas.ChatResponse`) for ``user_input``."""
        processed_input = preprocess(user_input)

        canned_reply = rule_based_reply(processed_input)
        if canned_reply:
            return canned_reply

        if not processed_input:
            # The message contained only punctuation/whitespace once
            # normalized (e.g. "???") - there's nothing meaningful to run
            # semantic search on, so go straight to the low-confidence path.
            return self._low_confidence_response(user_input, best_score=0.0)

        best_index, best_score = self._best_match(processed_input)

        # If the literal input doesn't already match well, try expanding
        # known keywords (e.g. "cgpa" -> "minimum cgpa required for
        # placements") and use that match instead if it scores higher.
        # This keeps keyword shortcuts useful for short/ambiguous queries
        # without overriding a literal match that's already a great fit
        # (e.g. the dataset question "resume tips" typed verbatim).
        if best_score < settings.HIGH_CONFIDENCE:
            expanded_input = expand_keywords(processed_input)
            if expanded_input != processed_input:
                exp_index, exp_score = self._best_match(expanded_input)
                if exp_score > best_score:
                    print(f"🔑 Keyword shortcut: '{processed_input}' -> '{expanded_input}'")
                    best_index, best_score = exp_index, exp_score

        matched_question = str(self.df["question"].iloc[best_index])
        matched_answer = self.answers[best_index]

        print(f"   ↳ matched '{matched_question}' (confidence={best_score:.3f})")

        if best_score >= settings.HIGH_CONFIDENCE:
            return {
                "reply": matched_answer,
                "confidence": round(best_score, 3),
                "source": "dataset",
                "matched_question": matched_question,
            }

        if best_score >= settings.MEDIUM_CONFIDENCE:
            return self._medium_confidence_response(
                user_input, matched_question, matched_answer, best_score
            )

        return self._low_confidence_response(user_input, best_score)

    def _best_match(self, processed_input: str) -> Tuple[int, float]:
        """Return ``(index, similarity)`` of the dataset question closest to ``processed_input``."""
        query_embedding = self.model.encode([processed_input])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        best_index = int(np.argmax(similarities))
        return best_index, float(similarities[best_index])

    def _medium_confidence_response(
        self, user_input: str, matched_question: str, matched_answer: str, best_score: float
    ) -> dict:
        """Ask Ollama (grounded in the matched dataset answer), or fall back to it directly."""
        if self.ollama.enabled and self.ollama_available:
            reply = self.ollama.generate(user_input, context=matched_answer)
            if reply:
                return {
                    "reply": reply,
                    "confidence": round(best_score, 3),
                    "source": "ollama-hybrid",
                    "context_used": matched_question,
                }

        return {
            "reply": matched_answer,
            "confidence": round(best_score, 3),
            "source": "dataset-fallback",
            "matched_question": matched_question,
        }

    def _low_confidence_response(self, user_input: str, best_score: float) -> dict:
        """Ask Ollama with no dataset context, or return a generic fallback message."""
        if self.ollama.enabled and self.ollama_available:
            reply = self.ollama.generate(user_input)
            if reply:
                return {
                    "reply": reply,
                    "confidence": round(best_score, 3),
                    "source": "ollama-pure",
                }

        return {
            "reply": FALLBACK_REPLY,
            "confidence": round(best_score, 3),
            "source": "fallback",
        }

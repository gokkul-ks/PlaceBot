"""Unit tests for the rule-based text-processing helpers in `text_utils`."""
from text_utils import expand_keywords, preprocess, rule_based_reply


def test_preprocess_lowercases_strips_and_removes_punctuation():
    assert preprocess("  How To Build a RESUME?!  ") == "how to build a resume"


def test_preprocess_handles_punctuation_only_input():
    assert preprocess("???") == ""
    assert preprocess("") == ""


def test_rule_based_reply_for_greeting():
    reply = rule_based_reply("hello")
    assert reply is not None
    assert reply["source"] == "greeting"


def test_rule_based_reply_for_thanks():
    reply = rule_based_reply("thanks a lot")
    assert reply is not None
    assert reply["source"] == "thanks"


def test_rule_based_reply_for_goodbye():
    reply = rule_based_reply("ok bye")
    assert reply is not None
    assert reply["source"] == "bye"


def test_rule_based_reply_returns_none_for_normal_question():
    assert rule_based_reply("how to prepare for placements") is None


def test_expand_keywords_on_short_query():
    assert expand_keywords("cgpa") == "minimum cgpa required for placements"
    assert expand_keywords("what is cgpa") == "minimum cgpa required for placements"


def test_expand_keywords_does_not_hijack_longer_questions():
    # "company" appears in this longer question, but it's specific enough
    # on its own and shouldn't be overridden by the keyword shortcut.
    query = "difference between service and product company"
    assert expand_keywords(query) == query


def test_expand_keywords_matches_whole_words_only():
    # "hr" must not match inside unrelated words such as "three".
    query = "i have three more questions"
    assert expand_keywords(query) == query


def test_expand_keywords_is_a_naive_shortcut_only():
    """
    `expand_keywords` has no knowledge of the dataset, so it will expand a
    short query even if that query is *itself* an exact dataset question
    (e.g. "resume tips" -> "how to build resume"). `PlacementChatbot` only
    uses this expansion as a fallback when the literal input doesn't
    already score well via semantic search (see `chatbot.get_response`),
    so an exact match like "resume tips" is answered correctly.
    """
    assert expand_keywords("resume tips") == "how to build resume"

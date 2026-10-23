import pytest

from smart_text_intelligence.preprocessing.remove_stop_words import (
    remove_stop_words,
)


def test_remove_stop_words_preserves_negation_by_default() -> None:
    assert remove_stop_words(
        ["this", "is", "not", "good", "never", "bad"],
        stop_words={"this", "is", "not", "good", "never", "bad"},
    ) == ["not", "never"]


def test_remove_stop_words_can_remove_negation_when_requested() -> None:
    assert remove_stop_words(
        ["not", "good"],
        stop_words={"not", "good"},
        preserve_negation=False,
    ) == []


def test_remove_stop_words_uses_case_insensitive_matching() -> None:
    assert remove_stop_words(["The", "cat"], stop_words={"the"}) == ["cat"]

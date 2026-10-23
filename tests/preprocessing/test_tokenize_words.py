import pytest

from smart_text_intelligence.preprocessing.tokenize_words import tokenize_words


def test_tokenize_words_keeps_contractions_and_numbers() -> None:
    assert tokenize_words("I can't pay 12.50!") == ["i", "can't", "pay", "12.50"]


def test_tokenize_words_can_preserve_case() -> None:
    assert tokenize_words("Hello WORLD", lowercase=False) == ["Hello", "WORLD"]


def test_tokenize_words_handles_unicode_apostrophes_and_mixed_punctuation() -> None:
    assert tokenize_words("Café’s naïve NLP—tools, 3.14!") == [
        "café’s",
        "naïve",
        "nlp",
        "tools",
        "3.14",
    ]


def test_tokenize_words_returns_empty_list_for_punctuation_only_input() -> None:
    assert tokenize_words("... — !!!") == []


def test_tokenize_words_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_words(None)  # type: ignore[arg-type]

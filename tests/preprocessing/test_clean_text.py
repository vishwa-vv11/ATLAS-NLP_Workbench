import pytest

from smart_text_intelligence.preprocessing.clean_text import clean_text


def test_clean_text_normalizes_and_preserves_sentence_punctuation() -> None:
    assert clean_text("  HELLO,   NLP! #text ") == "hello, nlp! text"


def test_clean_text_can_preserve_case_and_remove_punctuation() -> None:
    assert clean_text(
        "Hello, NLP!",
        lowercase=False,
        preserve_sentence_punctuation=False,
    ) == "Hello NLP"


def test_clean_text_can_remove_apostrophes() -> None:
    assert clean_text("isn't", preserve_apostrophes=False) == "isn t"


def test_clean_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(None)  # type: ignore[arg-type]

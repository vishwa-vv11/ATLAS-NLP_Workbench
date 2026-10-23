import pytest

from smart_text_intelligence.preprocessing.split_sentences import split_sentences


def test_split_sentences_splits_on_sentence_punctuation() -> None:
    assert split_sentences("First sentence. Is it ready? Yes!") == [
        "First sentence.",
        "Is it ready?",
        "Yes!",
    ]


def test_split_sentences_handles_empty_and_whitespace_input() -> None:
    assert split_sentences("  ") == []


def test_split_sentences_does_not_split_without_following_whitespace() -> None:
    assert split_sentences("Wait...Next") == ["Wait...Next"]


def test_split_sentences_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        split_sentences(None)  # type: ignore[arg-type]

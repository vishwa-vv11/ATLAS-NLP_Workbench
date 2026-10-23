import pytest

from smart_text_intelligence.preprocessing.lemmatize_tokens import (
    lemmatize_tokens,
)


def test_lemmatize_tokens_uses_noun_part_of_speech_by_default() -> None:
    assert lemmatize_tokens(["cars", "mice"]) == ["car", "mouse"]


def test_lemmatize_tokens_accepts_a_verb_part_of_speech() -> None:
    assert lemmatize_tokens(["running"], part_of_speech="v") == ["run"]


def test_lemmatize_tokens_rejects_invalid_part_of_speech() -> None:
    with pytest.raises(ValueError, match="WordNet POS"):
        lemmatize_tokens(["running"], part_of_speech="x")  # type: ignore[arg-type]

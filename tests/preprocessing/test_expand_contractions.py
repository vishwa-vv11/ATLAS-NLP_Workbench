import pytest

from smart_text_intelligence.preprocessing.expand_contractions import (
    expand_contractions,
)


def test_expand_contractions_expands_negation_and_common_forms() -> None:
    assert expand_contractions("I can't; we're ready.") == "I cannot; we are ready."


def test_expand_contractions_supports_curly_apostrophes_and_case() -> None:
    text = "IT" + chr(0x2019) + "LL work; Won" + chr(0x2019) + "t stop."
    assert expand_contractions(text) == "IT WILL work; Will not stop."


def test_expand_contractions_leaves_ambiguous_and_unknown_forms_unchanged() -> None:
    assert expand_contractions("James's book; it's ready; y'all") == (
        "James's book; it's ready; you all"
    )


def test_expand_contractions_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        expand_contractions(None)  # type: ignore[arg-type]

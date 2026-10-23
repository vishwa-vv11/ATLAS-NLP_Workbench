import re


_WORD_PATTERN = re.compile(
    r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*|\d+(?:[.,]\d+)*",
    re.UNICODE,
)


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    """Return word-like tokens, keeping internal apostrophes in contractions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens = _WORD_PATTERN.findall(text)
    return [token.lower() for token in tokens] if lowercase else tokens

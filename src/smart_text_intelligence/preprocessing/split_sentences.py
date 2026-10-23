import re


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence-ending punctuation followed by whitespace.

    This lightweight rule-based splitter does not resolve abbreviations such as
    "Dr." and is intended for the project's introductory preprocessing stage.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(text.strip())
        if sentence.strip()
    ]

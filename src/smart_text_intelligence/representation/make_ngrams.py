from collections.abc import Sequence


def make_ngrams(tokens: Sequence[str], n: int = 2) -> list[tuple[str, ...]]:
    """Return all contiguous n-grams in token order."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]

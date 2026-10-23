from collections import Counter
from collections.abc import Sequence


def word_frequency(tokens: Sequence[str], *, top_n: int | None = None) -> dict[str, int]:
    """Count tokens and return them in descending-frequency, stable order."""
    if top_n is not None and top_n < 0:
        raise ValueError("top_n must not be negative")
    counts = Counter(tokens)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return dict(ordered)

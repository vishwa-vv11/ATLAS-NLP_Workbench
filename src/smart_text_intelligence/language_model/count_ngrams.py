from collections import Counter
from collections.abc import Sequence


def count_ngrams(
    sentences: Sequence[Sequence[str]],
    n: int = 2,
    *,
    start_token: str = "<s>",
    end_token: str = "</s>",
) -> Counter[tuple[str, ...]]:
    """Count n-grams, adding n-1 start markers and one end marker per sentence."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    counts: Counter[tuple[str, ...]] = Counter()
    for sentence in sentences:
        padded = [start_token] * (n - 1) + list(sentence) + [end_token]
        counts.update(
            tuple(padded[index:index + n])
            for index in range(len(padded) - n + 1)
        )
    return counts

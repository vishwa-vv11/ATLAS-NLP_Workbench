from collections import Counter
from collections.abc import Iterable

from smart_text_intelligence.language_model.bigram_probability import (
    bigram_probability,
)


def smoothed_bigram_probabilities(
    bigram_counts: Counter[tuple[str, str]],
    vocabulary: Iterable[str],
    *,
    alpha: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Return add-alpha probabilities for every context/token pair."""
    if alpha <= 0:
        raise ValueError("alpha must be positive for smoothed probabilities")
    words = list(dict.fromkeys(vocabulary))
    if not words:
        raise ValueError("vocabulary must not be empty")
    contexts = list(dict.fromkeys(previous for previous, _ in bigram_counts))
    return {
        (context, token): bigram_probability(
            bigram_counts,
            context,
            token,
            vocabulary_size=len(words),
            alpha=alpha,
        )
        for context in contexts
        for token in words
    }

import math
from collections import Counter
from collections.abc import Sequence

from smart_text_intelligence.language_model.bigram_probability import (
    bigram_probability,
)


def calculate_perplexity(
    sentences: Sequence[Sequence[str]],
    bigram_counts: Counter[tuple[str, str]],
    *,
    vocabulary: set[str],
    alpha: float = 0.0,
    unknown_token: str = "<UNK>",
    start_token: str = "<s>",
    end_token: str = "</s>",
) -> float:
    """Calculate held-out perplexity from bigram counts and optional add-alpha."""
    if not sentences:
        raise ValueError("at least one sentence is required")
    if not vocabulary:
        raise ValueError("vocabulary must not be empty")
    if alpha < 0:
        raise ValueError("alpha must not be negative")

    evaluation_vocabulary = set(vocabulary) | {end_token, unknown_token}
    log_probability = 0.0
    token_count = 0
    for sentence in sentences:
        words = [
            token if token in vocabulary else unknown_token
            for token in sentence
        ]
        padded = [start_token] + words + [end_token]
        for previous, token in zip(padded, padded[1:]):
            probability = bigram_probability(
                bigram_counts,
                previous,
                token,
                vocabulary_size=len(evaluation_vocabulary),
                alpha=alpha,
            )
            if probability <= 0:
                return math.inf
            log_probability += math.log(probability)
            token_count += 1
    return math.exp(-log_probability / token_count)

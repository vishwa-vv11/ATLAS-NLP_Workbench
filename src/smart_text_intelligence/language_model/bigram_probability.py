from collections import Counter


def bigram_probability(
    bigram_counts: Counter[tuple[str, str]],
    previous_token: str,
    token: str,
    *,
    vocabulary_size: int,
    alpha: float = 0.0,
) -> float:
    """Calculate an MLE or add-alpha conditional bigram probability."""
    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be positive")
    if alpha < 0:
        raise ValueError("alpha must not be negative")
    numerator = bigram_counts[(previous_token, token)] + alpha
    denominator = sum(
        count
        for (context, _), count in bigram_counts.items()
        if context == previous_token
    ) + alpha * vocabulary_size
    if denominator == 0:
        if alpha == 0:
            return 0.0
        raise ValueError("cannot smooth a context absent from the bigram counts")
    return numerator / denominator

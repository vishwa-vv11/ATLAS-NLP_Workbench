from collections.abc import Sequence
from typing import Any, Literal

from gensim.models import Word2Vec


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    sg: Literal[0, 1] = 0,
    epochs: int = 20,
    seed: int = 42,
) -> Any:
    """Train a reproducible Gensim Word2Vec model (CBOW by default)."""
    if not sentences or not any(sentences):
        raise ValueError("at least one non-empty tokenized sentence is required")
    if vector_size < 1 or window < 1 or min_count < 1 or epochs < 1:
        raise ValueError("vector_size, window, min_count, and epochs must be positive")
    if sg not in {0, 1}:
        raise ValueError("sg must be 0 (CBOW) or 1 (Skip-gram)")
    return Word2Vec(
        sentences=[list(sentence) for sentence in sentences],
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,
        seed=seed,
        epochs=epochs,
    )

from collections.abc import Sequence
from typing import Any

from gensim.models import FastText


def train_fasttext(
    sentences: Sequence[Sequence[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 20,
    seed: int = 42,
) -> Any:
    """Train Gensim FastText; character n-grams can provide vectors for OOV words."""
    if not sentences or not any(sentences):
        raise ValueError("at least one non-empty tokenized sentence is required")
    if vector_size < 1 or window < 1 or min_count < 1 or epochs < 1:
        raise ValueError("vector_size, window, min_count, and epochs must be positive")
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=seed,
    )
    model.build_vocab([list(sentence) for sentence in sentences])
    model.train(
        [list(sentence) for sentence in sentences],
        total_examples=model.corpus_count,
        epochs=epochs,
    )
    return model

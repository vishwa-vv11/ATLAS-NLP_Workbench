from collections import Counter
from collections.abc import Sequence

import numpy as np


def train_glove(
    sentences: Sequence[Sequence[str]],
    *,
    vector_size: int = 50,
    window: int = 5,
    epochs: int = 50,
    learning_rate: float = 0.05,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Train a compact educational GloVe model with weighted least squares.

    This implementation favors clarity over speed and is intended for small
    corpora. Use a published, licensed pretrained model for production-scale
    embeddings.
    """
    if not sentences or not any(sentences):
        raise ValueError("at least one non-empty tokenized sentence is required")
    if vector_size < 1 or window < 1 or epochs < 1:
        raise ValueError("vector_size, window, and epochs must be positive")
    if learning_rate <= 0 or x_max <= 0 or alpha <= 0:
        raise ValueError("learning_rate, x_max, and alpha must be positive")

    vocabulary = list(dict.fromkeys(token for sentence in sentences for token in sentence))
    if not vocabulary:
        raise ValueError("corpus contains no tokens")
    indices = {word: index for index, word in enumerate(vocabulary)}
    cooccurrences: Counter[tuple[int, int]] = Counter()
    for sentence in sentences:
        for index, word in enumerate(sentence):
            for context_index in range(
                max(0, index - window),
                min(len(sentence), index + window + 1),
            ):
                if context_index == index:
                    continue
                context = sentence[context_index]
                cooccurrences[(indices[word], indices[context])] += (
                    1.0 / abs(index - context_index)
                )

    generator = np.random.default_rng(seed)
    word_vectors = generator.normal(0, 0.1, (len(vocabulary), vector_size))
    context_vectors = generator.normal(0, 0.1, (len(vocabulary), vector_size))
    word_bias = np.zeros(len(vocabulary))
    context_bias = np.zeros(len(vocabulary))
    entries = list(cooccurrences.items())

    for _ in range(epochs):
        generator.shuffle(entries)
        for (word_index, context_index), count in entries:
            weight = min(1.0, (count / x_max) ** alpha)
            error = (
                np.dot(word_vectors[word_index], context_vectors[context_index])
                + word_bias[word_index]
                + context_bias[context_index]
                - np.log(count)
            )
            gradient = 2.0 * weight * error
            old_word_vector = word_vectors[word_index].copy()
            word_vectors[word_index] -= (
                learning_rate * gradient * context_vectors[context_index]
            )
            context_vectors[context_index] -= (
                learning_rate * gradient * old_word_vector
            )
            word_bias[word_index] -= learning_rate * gradient
            context_bias[context_index] -= learning_rate * gradient

    combined = word_vectors + context_vectors
    return {
        word: combined[index].astype(np.float32)
        for index, word in enumerate(vocabulary)
    }

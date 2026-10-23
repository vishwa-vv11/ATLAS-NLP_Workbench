from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _get_vector(model: Any, word: str) -> np.ndarray:
    keyed_vectors = getattr(model, "wv", model)
    try:
        vector = keyed_vectors[word]
    except (KeyError, TypeError) as error:
        raise KeyError(f"word is not in the embedding vocabulary: {word}") from error
    return np.asarray(vector, dtype=float)


def vector_arithmetic(
    model: Any,
    *,
    positive: Sequence[str],
    negative: Sequence[str] = (),
) -> list[float]:
    """Compute the mean positive vector minus the mean negative vector."""
    if not positive:
        raise ValueError("at least one positive word is required")
    result = np.mean([_get_vector(model, word) for word in positive], axis=0)
    if negative:
        result = result - np.mean(
            [_get_vector(model, word) for word in negative],
            axis=0,
        )
    return result.tolist()


def nearest_words(
    model: Any,
    vector: Sequence[float],
    *,
    top_n: int = 5,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Find nearest known vectors to a query vector by cosine similarity."""
    if top_n < 0:
        raise ValueError("top_n must not be negative")
    keyed_vectors: Mapping[str, Any] = getattr(model, "wv", model)
    query = np.asarray(vector, dtype=float)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        raise ValueError("query vector must not be zero")
    excluded = exclude or set()
    ranked: list[tuple[str, float]] = []
    for word, candidate in keyed_vectors.items():
        candidate_array = np.asarray(candidate, dtype=float)
        if word in excluded:
            continue
        candidate_norm = float(np.linalg.norm(candidate_array))
        if candidate_norm:
            score = float(np.dot(query, candidate_array) / (query_norm * candidate_norm))
            ranked.append((word, score))
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:top_n]

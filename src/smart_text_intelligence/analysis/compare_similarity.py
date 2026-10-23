from collections.abc import Mapping
from typing import Any

from smart_text_intelligence.embeddings.cosine_similarity import cosine_similarity


def compare_similarity(
    model: Any,
    word: str,
    *,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return vocabulary items ranked by cosine similarity to a query word."""
    if top_n < 0:
        raise ValueError("top_n must not be negative")
    vectors: Mapping[str, Any] = getattr(model, "wv", model)
    if hasattr(vectors, "most_similar"):
        return [
            (str(token), float(score))
            for token, score in vectors.most_similar(word, topn=top_n)
        ]

    if word not in vectors:
        raise KeyError(f"word is not in the embedding vocabulary: {word}")
    ranked = [
        (token, cosine_similarity(vectors[word], vector))
        for token, vector in vectors.items()
        if token != word
    ]
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:top_n]

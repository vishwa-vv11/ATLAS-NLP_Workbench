from collections.abc import Sequence
import math


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """Return cosine similarity, rejecting incompatible or zero-length vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same number of dimensions")
    if len(vector_a) == 0:
        raise ValueError("vectors must not be empty")
    values_a = [float(value) for value in vector_a]
    values_b = [float(value) for value in vector_b]
    dot = sum(left * right for left, right in zip(values_a, values_b))
    norm_a = math.sqrt(sum(value ** 2 for value in values_a))
    norm_b = math.sqrt(sum(value ** 2 for value in values_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot / (norm_a * norm_b)

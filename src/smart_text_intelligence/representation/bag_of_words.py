from collections import Counter
from collections.abc import Sequence

from smart_text_intelligence.representation.one_hot import build_vocabulary


def bag_of_words(
    documents: Sequence[Sequence[str]],
) -> tuple[list[str], list[list[int]]]:
    """Return first-seen vocabulary and document-term count matrix."""
    vocabulary = build_vocabulary(documents)
    matrix = [
        [Counter(document).get(token, 0) for token in vocabulary]
        for document in documents
    ]
    return vocabulary, matrix

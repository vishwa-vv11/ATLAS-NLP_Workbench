from collections.abc import Sequence


def build_vocabulary(documents: Sequence[Sequence[str]]) -> list[str]:
    """Build a deterministic vocabulary in first-seen order."""
    return list(dict.fromkeys(token for document in documents for token in document))


def encode_one_hot(
    tokens: Sequence[str],
    vocabulary: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Encode each token as a vocabulary-length one-hot vector."""
    selected_vocabulary = list(vocabulary or dict.fromkeys(tokens))
    if len(selected_vocabulary) != len(set(selected_vocabulary)):
        raise ValueError("vocabulary must not contain duplicate tokens")
    indices = {token: index for index, token in enumerate(selected_vocabulary)}
    return {
        token: [
            int(index == indices[token])
            for index in range(len(selected_vocabulary))
        ]
        for token in tokens
        if token in indices
    }

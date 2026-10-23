from pathlib import Path

import numpy as np


def load_glove_embeddings(
    path: str | Path,
    *,
    max_words: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a plain-text GloVe file containing one word and its vector per line."""
    glove_path = Path(path)
    if not glove_path.is_file():
        raise FileNotFoundError(f"GloVe file does not exist: {glove_path}")
    if max_words is not None and max_words < 1:
        raise ValueError("max_words must be positive")

    vectors: dict[str, np.ndarray] = {}
    vector_size: int | None = None
    with glove_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            word, *values = parts
            if vector_size is None:
                vector_size = len(values)
                if vector_size == 0:
                    raise ValueError(f"no vector values on line {line_number}")
            if len(values) != vector_size:
                raise ValueError(
                    f"inconsistent vector size on GloVe line {line_number}"
                )
            if word in vectors:
                raise ValueError(f"duplicate GloVe word on line {line_number}: {word}")
            try:
                vectors[word] = np.asarray(values, dtype=np.float32)
            except ValueError as error:
                raise ValueError(
                    f"non-numeric GloVe vector on line {line_number}"
                ) from error
            if max_words is not None and len(vectors) >= max_words:
                break
    if not vectors:
        raise ValueError("GloVe file contains no word vectors")
    return vectors

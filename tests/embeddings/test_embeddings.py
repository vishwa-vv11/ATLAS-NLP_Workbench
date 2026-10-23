import numpy as np
import pytest

from smart_text_intelligence.analysis.compare_similarity import compare_similarity
from smart_text_intelligence.embeddings.cosine_similarity import cosine_similarity
from smart_text_intelligence.embeddings.load_glove import load_glove_embeddings
from smart_text_intelligence.embeddings.train_fasttext import train_fasttext
from smart_text_intelligence.embeddings.train_glove import train_glove
from smart_text_intelligence.embeddings.train_word2vec import train_word2vec
from smart_text_intelligence.embeddings.vector_arithmetic import (
    nearest_words,
    vector_arithmetic,
)


def test_cosine_similarity_calculates_known_value() -> None:
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(1 / np.sqrt(2))


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity([0, 0], [1, 1])


def test_glove_loader_parses_vectors_and_validates_dimensions(tmp_path) -> None:
    path = tmp_path / "vectors.txt"
    path.write_text("cat 1.0 0.0\ndog 0.0 1.0\n", encoding="utf-8")
    vectors = load_glove_embeddings(path)
    assert vectors["cat"].tolist() == [1.0, 0.0]

    path.write_text("cat 1.0 0.0\ndog 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_glove_embeddings(path)


def test_glove_training_returns_finite_vectors() -> None:
    vectors = train_glove(
        [["cats", "purr"], ["dogs", "bark"], ["cats", "sleep"]],
        vector_size=4,
        window=2,
        epochs=3,
        seed=1,
    )
    assert set(vectors) == {"cats", "purr", "dogs", "bark", "sleep"}
    assert all(np.isfinite(vector).all() for vector in vectors.values())


def test_word2vec_supports_cbow_and_skipgram() -> None:
    sentences = [["cats", "purr"], ["dogs", "bark"], ["cats", "sleep"]]
    cbow = train_word2vec(sentences, vector_size=8, sg=0, epochs=2)
    skipgram = train_word2vec(sentences, vector_size=8, sg=1, epochs=2)
    assert cbow.wv.vector_size == skipgram.wv.vector_size == 8


def test_fasttext_generates_vectors_for_oov_words() -> None:
    model = train_fasttext(
        [["play", "playing"], ["player", "plays"]],
        vector_size=8,
        epochs=2,
    )
    assert np.isfinite(model.wv["playful"]).all()


def test_vector_arithmetic_and_similarity_work_with_mapping() -> None:
    vectors = {
        "king": np.array([1.0, 1.0]),
        "man": np.array([1.0, 0.0]),
        "woman": np.array([0.0, 1.0]),
        "queen": np.array([0.0, 2.0]),
    }
    result = vector_arithmetic(vectors, positive=["king", "woman"], negative=["man"])
    assert result == pytest.approx([-0.5, 1.0])
    assert nearest_words(vectors, result, top_n=1)[0][0] == "queen"
    assert compare_similarity(vectors, "king", top_n=1)[0][0] in vectors

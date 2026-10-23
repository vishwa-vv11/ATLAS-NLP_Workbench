from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from smart_text_intelligence.embeddings.train_word2vec import train_word2vec
from smart_text_intelligence.evaluation.classification_metrics import (
    classification_metrics,
)
from smart_text_intelligence.preprocessing.tokenize_words import tokenize_words


def _average_vectors(texts: Sequence[str], model: Any) -> np.ndarray:
    vectors = np.zeros((len(texts), model.wv.vector_size), dtype=np.float32)
    for row, text in enumerate(texts):
        known = [
            model.wv[token]
            for token in tokenize_words(text)
            if token in model.wv.key_to_index
        ]
        if known:
            vectors[row] = np.mean(known, axis=0)
    return vectors


def compare_representations(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    train_labels: Sequence[str],
    test_labels: Sequence[str],
    *,
    include_embeddings: bool = True,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compare BoW, TF-IDF, and optional averaged Word2Vec features.

    All vectorizers and the embedding model are fitted using training text only.
    Results are dataset-specific and should not be interpreted as universal ranks.
    """
    if not train_texts or not test_texts:
        raise ValueError("training and test text must both be non-empty")
    if len(train_texts) != len(train_labels):
        raise ValueError("training text and labels must have equal lengths")
    if len(test_texts) != len(test_labels):
        raise ValueError("test text and labels must have equal lengths")
    if len(set(train_labels)) < 2:
        raise ValueError("training data must contain at least two label classes")

    results: dict[str, dict[str, Any]] = {}
    for name, vectorizer in (
        ("bow", CountVectorizer()),
        ("tfidf", TfidfVectorizer()),
    ):
        pipeline = make_pipeline(
            vectorizer,
            LogisticRegression(max_iter=1000, random_state=random_state),
        )
        pipeline.fit(list(train_texts), list(train_labels))
        predictions = pipeline.predict(list(test_texts))
        results[name] = classification_metrics(test_labels, predictions)

    if include_embeddings:
        sentences = [tokenize_words(text) for text in train_texts]
        embedding_model = train_word2vec(
            sentences,
            vector_size=50,
            min_count=1,
            epochs=30,
            seed=random_state,
        )
        train_vectors = _average_vectors(train_texts, embedding_model)
        test_vectors = _average_vectors(test_texts, embedding_model)
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ).fit(train_vectors, train_labels)
        results["word2vec_average"] = classification_metrics(
            test_labels,
            classifier.predict(test_vectors),
        )
    return results

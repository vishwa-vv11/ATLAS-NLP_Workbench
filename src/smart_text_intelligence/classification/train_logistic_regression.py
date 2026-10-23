from typing import Literal

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_logistic_regression(
    texts: list[str],
    labels: list[str],
    *,
    representation: Literal["bow", "tfidf"] = "tfidf",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Fit a Logistic Regression text pipeline on training examples."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and have equal lengths")
    if representation not in {"bow", "tfidf"}:
        raise ValueError("representation must be 'bow' or 'tfidf'")
    vectorizer = (
        CountVectorizer() if representation == "bow" else TfidfVectorizer()
    )
    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline(
        [("features", vectorizer), ("classifier", classifier)]
    ).fit(texts, labels)

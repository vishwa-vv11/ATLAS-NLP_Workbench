from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(
    texts: list[str],
    labels: list[str],
    *,
    representation: Literal["bow", "tfidf"] = "tfidf",
    alpha: float = 1.0,
) -> Pipeline:
    """Fit a Multinomial Naive Bayes text pipeline on training examples."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and have equal lengths")
    vectorizer = (
        CountVectorizer() if representation == "bow" else TfidfVectorizer()
    )
    if representation not in {"bow", "tfidf"}:
        raise ValueError("representation must be 'bow' or 'tfidf'")
    return Pipeline(
        [("features", vectorizer), ("classifier", MultinomialNB(alpha=alpha))]
    ).fit(texts, labels)

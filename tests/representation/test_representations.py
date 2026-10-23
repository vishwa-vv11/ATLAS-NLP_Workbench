import math

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from smart_text_intelligence.representation.bag_of_words import bag_of_words
from smart_text_intelligence.representation.make_ngrams import make_ngrams
from smart_text_intelligence.representation.one_hot import (
    build_vocabulary,
    encode_one_hot,
)
from smart_text_intelligence.representation.tfidf import ManualTfidf


def test_one_hot_encoding_and_first_seen_vocabulary() -> None:
    documents = [["red", "blue"], ["red", "green"]]
    vocabulary = build_vocabulary(documents)
    assert vocabulary == ["red", "blue", "green"]
    assert encode_one_hot(["blue", "unknown"], vocabulary) == {
        "blue": [0, 1, 0]
    }


def test_bag_of_words_returns_document_term_counts() -> None:
    assert bag_of_words([["a", "b", "a"], ["b"]]) == (
        ["a", "b"],
        [[2, 1], [0, 1]],
    )


def test_make_ngrams_creates_contiguous_windows() -> None:
    assert make_ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert make_ngrams(["a", "b"], 3) == []


def test_make_ngrams_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_ngrams(["a"], 0)


def test_make_ngrams_handles_empty_and_single_token_inputs() -> None:
    assert make_ngrams([], 2) == []
    assert make_ngrams(["only"], 1) == [("only",)]
    assert make_ngrams(["a", "b"], 2) == [("a", "b")]


def test_make_ngrams_preserves_repeated_token_windows() -> None:
    assert make_ngrams(["a", "a", "b", "a"], 2) == [
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
    ]


def test_manual_tfidf_unsmoothed_idf_matches_formula() -> None:
    documents = [["cat", "cat", "dog"], ["cat", "bird"]]
    vectorizer = ManualTfidf()
    result = vectorizer.fit_transform(documents)
    assert vectorizer.idf_["cat"] == 0
    assert math.isclose(vectorizer.idf_["dog"], math.log(2))
    assert result[0][vectorizer.vocabulary_.index("dog")] > 0


def test_manual_tfidf_smoothed_values_match_hand_calculated_formula() -> None:
    documents = [["cat", "cat", "dog"], ["cat", "bird"]]
    vectorizer = ManualTfidf(smooth_idf=True, norm="l2")
    result = vectorizer.fit_transform(documents)

    cat_idf = 1.0
    rare_idf = math.log(3 / 2) + 1
    first_norm = math.sqrt((2 * cat_idf) ** 2 + rare_idf**2)
    second_norm = math.sqrt(cat_idf**2 + rare_idf**2)

    assert vectorizer.vocabulary_ == ["cat", "dog", "bird"]
    assert vectorizer.idf_ == pytest.approx(
        {"cat": cat_idf, "dog": rare_idf, "bird": rare_idf}
    )
    assert result[0] == pytest.approx(
        [2 * cat_idf / first_norm, rare_idf / first_norm, 0]
    )
    assert result[1] == pytest.approx(
        [cat_idf / second_norm, 0, rare_idf / second_norm]
    )


def test_manual_tfidf_transform_maps_unknown_terms_to_zero_features() -> None:
    vectorizer = ManualTfidf(smooth_idf=True, norm="l2").fit([["cat"]])

    assert vectorizer.transform([["unknown"], []]) == [[0.0], [0.0]]


def test_manual_tfidf_smoothed_matches_sklearn_convention() -> None:
    documents = ["cat cat dog", "cat bird"]
    tokenized = [document.split() for document in documents]
    manual = ManualTfidf(smooth_idf=True, norm="l2")
    result = manual.fit_transform(tokenized)
    sklearn_result = TfidfVectorizer().fit_transform(documents).toarray()
    sklearn_vocab = TfidfVectorizer().fit(documents).vocabulary_
    indices = [manual.vocabulary_.index(word) for word in sorted(sklearn_vocab)]
    reordered = [[row[index] for index in indices] for row in result]
    for manual_row, sklearn_row in zip(reordered, sklearn_result):
        assert manual_row == pytest.approx(sklearn_row.tolist())


def test_manual_tfidf_transform_requires_fit_and_supports_empty_documents() -> None:
    transformer = ManualTfidf()
    with pytest.raises(ValueError, match="fit"):
        transformer.transform([["cat"]])
    assert ManualTfidf().fit_transform([[], []]) == [[], []]


def test_manual_tfidf_rejects_invalid_normalization() -> None:
    with pytest.raises(ValueError, match="norm"):
        ManualTfidf(norm="l3").fit([["cat"]])

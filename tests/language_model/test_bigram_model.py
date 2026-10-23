from collections import Counter
import math

import pytest

from smart_text_intelligence.language_model.bigram_model import BigramLanguageModel
from smart_text_intelligence.language_model.bigram_probability import (
    bigram_probability,
)
from smart_text_intelligence.language_model.calculate_perplexity import (
    calculate_perplexity,
)
from smart_text_intelligence.language_model.count_ngrams import count_ngrams
from smart_text_intelligence.language_model.smooth_probabilities import (
    smoothed_bigram_probabilities,
)


def test_count_ngrams_adds_sentence_boundary_markers() -> None:
    assert count_ngrams([["i", "run"]], 2) == Counter(
        {("<s>", "i"): 1, ("i", "run"): 1, ("run", "</s>"): 1}
    )


def test_bigram_probability_computes_mle_and_laplace() -> None:
    counts = Counter({("a", "b"): 2, ("a", "c"): 1})
    assert bigram_probability(counts, "a", "b", vocabulary_size=2) == pytest.approx(
        2 / 3
    )
    assert bigram_probability(
        counts,
        "a",
        "b",
        vocabulary_size=2,
        alpha=1,
    ) == pytest.approx(3 / 5)


def test_bigram_probability_is_zero_for_unseen_pair_without_smoothing() -> None:
    counts = Counter({("a", "b"): 2, ("a", "c"): 1})

    assert bigram_probability(
        counts,
        "a",
        "missing",
        vocabulary_size=3,
    ) == 0


def test_bigram_probability_rejects_invalid_smoothing_parameters() -> None:
    counts = Counter({("a", "b"): 1})
    with pytest.raises(ValueError, match="vocabulary_size"):
        bigram_probability(counts, "a", "b", vocabulary_size=0)
    with pytest.raises(ValueError, match="alpha"):
        bigram_probability(counts, "a", "b", vocabulary_size=2, alpha=-1)


def test_smoothed_probabilities_sum_to_one_per_context() -> None:
    counts = Counter({("a", "b"): 2})
    result = smoothed_bigram_probabilities(counts, {"b", "c"}, alpha=1)
    assert result[("a", "b")] + result[("a", "c")] == pytest.approx(1)


def test_bigram_model_scores_known_and_unknown_words() -> None:
    model = BigramLanguageModel(alpha=1).fit([["we", "learn"], ["we", "read"]])
    assert model.probability("<s>", "we") > 0
    assert model.probability("we", "unknown") > 0
    assert math.isfinite(model.perplexity([["we", "learn"]]))


def test_bigram_model_maps_unknown_context_and_token_to_unknown_token() -> None:
    model = BigramLanguageModel(alpha=1).fit([["we", "learn"], ["we", "read"]])

    assert model.probability("unseen-context", "unseen-token") == pytest.approx(
        model.probability("<UNK>", "<UNK>")
    )
    assert math.isfinite(model.perplexity([["brand-new", "words"]]))


def test_bigram_model_includes_sentence_boundaries_and_empty_sentences() -> None:
    model = BigramLanguageModel(alpha=1).fit([["hello"], []])

    assert model.counts[("<s>", "hello")] == 1
    assert model.counts[("hello", "</s>")] == 1
    assert model.counts[("<s>", "</s>")] == 1
    assert math.isfinite(model.perplexity([[]]))


def test_unsmoothed_perplexity_is_infinite_for_unseen_bigram() -> None:
    counts = count_ngrams([["a", "b"]], n=2)
    score = calculate_perplexity(
        [["b", "a"]],
        counts,
        vocabulary={"a", "b"},
    )
    assert math.isinf(score)

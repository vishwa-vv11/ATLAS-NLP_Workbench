from smart_text_intelligence.analysis.extract_keywords import extract_keywords
from smart_text_intelligence.analysis.sentiment_baseline import sentiment_baseline
from smart_text_intelligence.analysis.word_frequency import word_frequency
from smart_text_intelligence.main import analyze_text, evaluate_language_model


def test_word_frequency_is_stable_and_limits_results() -> None:
    assert word_frequency(["b", "a", "b", "a", "c"], top_n=2) == {
        "a": 2,
        "b": 2,
    }


def test_keyword_extraction_ranks_terms_and_handles_empty_text() -> None:
    assert extract_keywords("apple banana apple", top_n=1)[0][0] == "apple"
    assert extract_keywords("   ") == []
    assert all(
        token not in {"isn", "re", "this", "we"}
        for token, _ in extract_keywords("This isn't bad. We're learning.")
    )


def test_sentiment_baseline_applies_negation_to_next_sentiment_word() -> None:
    assert sentiment_baseline(["not", "very", "good"]) == {
        "label": "negative",
        "score": -1,
    }
    assert sentiment_baseline(["excellent", "service"])["label"] == "positive"


def test_analyze_text_returns_all_analysis_sections() -> None:
    results = analyze_text("This is not a bad example.")
    assert set(results) == {
        "preprocessing",
        "word_frequency",
        "keywords",
        "sentiment_baseline",
    }


def test_analyze_text_expands_contractions_before_sentiment_scoring() -> None:
    assert analyze_text("This isn't a bad example.")["sentiment_baseline"] == {
        "label": "positive",
        "score": 1,
    }


def test_language_model_evaluation_uses_separate_texts() -> None:
    results = evaluate_language_model(
        "cats purr. dogs bark.",
        "cats bark.",
    )
    assert results["perplexity"] > 0
    assert results["training_sentences"] == 2

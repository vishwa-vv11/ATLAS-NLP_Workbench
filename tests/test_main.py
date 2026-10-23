import math

import pytest

from smart_text_intelligence.main import (
    evaluate_language_model,
    preprocess_text,
    represent_texts,
)


def test_preprocessing_pipeline_preserves_negation() -> None:
    result = preprocess_text("This isn't good.")

    assert result["tokens"] == ["this", "is", "not", "good"]
    assert "not" in result["without_stop_words"]


def test_main_representation_workflow_calls_manual_features() -> None:
    result = represent_texts(["Cats purr.", "Dogs bark."])

    assert result["vocabulary"] == ["cats", "purr", "dogs", "bark"]
    assert result["bag_of_words"] == [[1, 1, 0, 0], [0, 0, 1, 1]]
    assert len(result["ngrams"]) == 2
    assert len(result["tfidf"]) == 2


def test_language_model_trace_exposes_each_held_out_bigram_probability() -> None:
    result = evaluate_language_model(
        "cats purr. dogs bark.",
        "cats unseen.",
    )
    trace = result["token_probabilities"]

    assert [(step["previous_token"], step["next_token"]) for step in trace] == [
        ("<s>", "cats"),
        ("cats", "<UNK>"),
        ("<UNK>", "</s>"),
    ]
    assert all(0 < step["probability"] <= 1 for step in trace)
    expected_perplexity = math.exp(
        -sum(math.log(step["probability"]) for step in trace) / len(trace)
    )
    assert result["perplexity"] == pytest.approx(expected_perplexity)

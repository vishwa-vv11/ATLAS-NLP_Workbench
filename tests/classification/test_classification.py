import pytest

from smart_text_intelligence.classification.predict_text import predict_text
from smart_text_intelligence.classification.train_logistic_regression import (
    train_logistic_regression,
)
from smart_text_intelligence.classification.train_naive_bayes import (
    train_naive_bayes,
)
from smart_text_intelligence.evaluation.classification_metrics import (
    classification_metrics,
)
from smart_text_intelligence.evaluation.error_analysis import error_analysis
from smart_text_intelligence.main import evaluate_classifier


TEXTS = [
    "win a prize now",
    "claim your cash prize",
    "free prize for you",
    "meeting at the office",
    "please review the report",
    "office meeting tomorrow",
]
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]


@pytest.mark.parametrize(
    "trainer",
    [train_naive_bayes, train_logistic_regression],
)
def test_classifiers_fit_and_predict(trainer) -> None:
    model = trainer(TEXTS, LABELS)
    predictions = predict_text(model, ["free prize", "office meeting"])
    assert predictions == ["spam", "ham"]


def test_classifier_evaluation_returns_holdout_metrics_and_errors() -> None:
    result = evaluate_classifier(TEXTS * 4, LABELS * 4, random_state=3)
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert len(result["predictions"]) == len(result["test_texts"])


def test_metrics_include_weighted_scores_and_confusion_matrix() -> None:
    result = classification_metrics(["ham", "spam"], ["ham", "ham"])
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]


def test_error_analysis_lists_only_misclassified_examples() -> None:
    assert error_analysis(["a", "b"], ["x", "y"], ["x", "x"]) == [
        {"text": "b", "true_label": "y", "predicted_label": "x"}
    ]


def test_classification_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="equal"):
        classification_metrics(["a"], [])

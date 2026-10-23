from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, Any]:
    """Return reproducible aggregate and per-class classification metrics."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have equal non-zero lengths")
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

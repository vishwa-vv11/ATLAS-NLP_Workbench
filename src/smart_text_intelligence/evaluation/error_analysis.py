from collections.abc import Sequence


def error_analysis(
    texts: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> list[dict[str, str]]:
    """List misclassified examples with their true and predicted labels."""
    if not (len(texts) == len(y_true) == len(y_pred)):
        raise ValueError("texts, y_true, and y_pred must have equal lengths")
    return [
        {"text": text, "true_label": true, "predicted_label": predicted}
        for text, true, predicted in zip(texts, y_true, y_pred)
        if true != predicted
    ]

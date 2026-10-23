from typing import Any


def predict_text(model: Any, texts: list[str]) -> list[str]:
    """Predict labels for input texts using a fitted scikit-learn estimator."""
    if not texts:
        return []
    if not hasattr(model, "predict"):
        raise TypeError("model must provide a predict method")
    return [str(label) for label in model.predict(texts)]

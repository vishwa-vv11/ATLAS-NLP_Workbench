from smart_text_intelligence.evaluation.compare_representations import (
    compare_representations,
)


def test_compare_representations_fits_methods_on_train_only() -> None:
    train_texts = [
        "prize claim offer",
        "free cash prize",
        "office meeting report",
        "meeting report please",
        "claim free offer",
        "office report meeting",
    ]
    train_labels = ["spam", "spam", "ham", "ham", "spam", "ham"]
    result = compare_representations(
        train_texts,
        ["prize offer", "office meeting"],
        train_labels,
        ["spam", "ham"],
        include_embeddings=True,
        random_state=4,
    )
    assert set(result) == {"bow", "tfidf", "word2vec_average"}
    assert all("f1_weighted" in metrics for metrics in result.values())


def test_compare_representations_rejects_single_class_training_data() -> None:
    import pytest

    with pytest.raises(ValueError, match="two label"):
        compare_representations(["a"], ["b"], ["one"], ["one"])

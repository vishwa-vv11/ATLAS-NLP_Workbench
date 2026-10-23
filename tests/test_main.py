from smart_text_intelligence.main import preprocess_text, represent_texts


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

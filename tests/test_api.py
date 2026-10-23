from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health_endpoint_reports_ready() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_returns_existing_nlp_pipeline_results() -> None:
    response = client.post(
        "/api/analyze",
        json={"text": "This isn't a bad example."},
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "preprocessing",
        "word_frequency",
        "keywords",
        "sentiment_baseline",
    }
    assert set(result["preprocessing"]) == {
        "expanded_text",
        "cleaned_text",
        "sentences",
        "tokens",
        "without_stop_words",
        "stemmed_tokens",
        "lemmatized_tokens",
    }
    assert "not" in result["preprocessing"]["without_stop_words"]
    assert result["sentiment_baseline"]["label"] == "positive"


def test_representation_endpoint_returns_manual_features() -> None:
    response = client.post(
        "/api/represent",
        json={"documents": ["cats purr", "dogs bark"]},
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "vocabulary",
        "one_hot_first_document",
        "bag_of_words",
        "ngrams",
        "tfidf_vocabulary",
        "tfidf",
    }
    assert result["bag_of_words"] == [[1, 1, 0, 0], [0, 0, 1, 1]]
    assert len(result["ngrams"]) == len(result["tfidf"]) == 2


def test_classification_endpoint_trains_and_returns_metrics() -> None:
    texts = [
        "win a prize now",
        "claim your cash prize",
        "free prize for you",
        "meeting at the office",
        "please review the report",
        "office meeting tomorrow",
        "win your prize",
        "office report meeting",
        "free cash now",
        "review office notes",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"] * 1 + [
        "spam",
        "ham",
        "spam",
        "ham",
    ]
    response = client.post(
        "/api/classify",
        json={"texts": texts, "labels": labels, "model": "naive_bayes"},
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "test_texts",
        "true_labels",
        "predictions",
        "metrics",
        "errors",
    }
    assert {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "labels",
        "confusion_matrix",
        "per_class",
    } <= set(result["metrics"])
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert len(result["predictions"]) == len(result["test_texts"])
    assert len(result["true_labels"]) == len(result["test_texts"])


def test_language_model_endpoint_returns_perplexity() -> None:
    response = client.post(
        "/api/language-model",
        json={
            "training_text": "cats purr. dogs bark.",
            "evaluation_text": "cats bark.",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "perplexity",
        "training_sentences",
        "evaluation_sentences",
        "token_probabilities",
    }
    assert result["perplexity"] > 0
    assert result["token_probabilities"]
    assert set(result["token_probabilities"][0]) == {
        "sentence",
        "previous_token",
        "next_token",
        "probability",
    }
    assert result["token_probabilities"][0]["previous_token"] == "<s>"
    assert result["token_probabilities"][0]["next_token"] == "cats"
    assert 0 < result["token_probabilities"][0]["probability"] <= 1
    assert result["token_probabilities"][-1]["next_token"] == "</s>"


def test_representation_comparison_endpoint_returns_all_sparse_methods() -> None:
    texts = [
        "win a prize now",
        "claim your cash prize",
        "free prize for you",
        "meeting at the office",
        "please review the report",
        "office meeting tomorrow",
        "win your prize",
        "office report meeting",
        "free cash now",
        "review office notes",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham", "spam", "ham", "spam", "ham"]
    response = client.post(
        "/api/compare",
        json={
            "texts": texts,
            "labels": labels,
            "include_embeddings": False,
        },
    )

    assert response.status_code == 200
    assert set(response.json()) == {"bow", "tfidf"}


def test_embedding_endpoint_trains_and_returns_similar_words() -> None:
    response = client.post(
        "/api/embeddings",
        json={
            "texts": [
                "cats purr and sleep",
                "cats sleep in the garden",
                "dogs bark and play",
                "dogs play in the garden",
            ],
            "query": "cats",
            "vector_size": 8,
            "top_n": 3,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result
    assert all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], float)
        for item in result
    )


def test_embedding_endpoint_returns_pca_projection_when_requested() -> None:
    response = client.post(
        "/api/embeddings",
        json={
            "texts": [
                "cats purr and sleep",
                "cats sleep in the garden",
                "dogs bark and play",
                "dogs play in the garden",
            ],
            "query": "CATS!",
            "vector_size": 8,
            "top_n": 3,
            "include_projection": True,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "cats"
    assert result["neighbors"]
    assert len(result["projection"]) >= 2
    assert all(
        set(point) == {"word", "x", "y"}
        and isinstance(point["word"], str)
        and isinstance(point["x"], float)
        and isinstance(point["y"], float)
        for point in result["projection"]
    )
    assert any(point["word"] == "cats" for point in result["projection"])


def test_invalid_embedding_query_is_reported_as_validation_error() -> None:
    response = client.post(
        "/api/embeddings",
        json={
            "texts": ["cats sleep", "dogs run"],
            "query": "two words",
            "include_projection": True,
        },
    )

    assert response.status_code == 422
    assert "exactly one word" in response.json()["detail"]


def test_invalid_requests_return_actionable_validation_errors() -> None:
    empty_text = client.post("/api/analyze", json={"text": ""})
    missing_text = client.post("/api/analyze", json={})
    empty_documents = client.post("/api/represent", json={"documents": []})
    mismatched_labels = client.post(
        "/api/classify",
        json={
            "texts": ["one", "two", "three", "four", "five"],
            "labels": ["a", "b", "a", "b"],
        },
    )
    empty_training_text = client.post(
        "/api/language-model",
        json={"training_text": "", "evaluation_text": "text"},
    )

    assert empty_text.status_code == 422
    assert missing_text.status_code == 422
    assert empty_documents.status_code == 422
    assert mismatched_labels.status_code == 422
    assert "same number" in mismatched_labels.json()["detail"]
    assert empty_training_text.status_code == 422


def test_independent_api_requests_can_run_concurrently() -> None:
    texts = [
        "Natural language has useful patterns.",
        "Text processing transforms words.",
        "Classical methods inspect text.",
        "Language models learn patterns.",
    ]

    def analyze(text: str) -> dict:
        with TestClient(app) as concurrent_client:
            response = concurrent_client.post("/api/analyze", json={"text": text})
            assert response.status_code == 200
            return response.json()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(analyze, texts))

    assert len(results) == len(texts)
    assert all(result["preprocessing"]["tokens"] for result in results)

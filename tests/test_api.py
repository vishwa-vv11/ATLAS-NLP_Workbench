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
    assert "not" in result["preprocessing"]["without_stop_words"]
    assert result["sentiment_baseline"]["label"] == "positive"


def test_representation_endpoint_returns_manual_features() -> None:
    response = client.post(
        "/api/represent",
        json={"documents": ["cats purr", "dogs bark"]},
    )

    assert response.status_code == 200
    assert response.json()["bag_of_words"] == [[1, 1, 0, 0], [0, 0, 1, 1]]


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
    assert 0 <= response.json()["metrics"]["accuracy"] <= 1
    assert len(response.json()["predictions"]) == len(response.json()["test_texts"])


def test_language_model_endpoint_returns_perplexity() -> None:
    response = client.post(
        "/api/language-model",
        json={
            "training_text": "cats purr. dogs bark.",
            "evaluation_text": "cats bark.",
        },
    )

    assert response.status_code == 200
    assert response.json()["perplexity"] > 0


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
    assert response.json()
    assert all(len(item) == 2 for item in response.json())


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
        isinstance(point["x"], float) and isinstance(point["y"], float)
        for point in result["projection"]
    )
    assert any(point["word"] == "cats" for point in result["projection"])


def test_invalid_requests_return_actionable_validation_errors() -> None:
    response = client.post("/api/analyze", json={"text": ""})

    assert response.status_code == 422

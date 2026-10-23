from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from smart_text_intelligence.main import (
    analyze_text,
    compare_dataset_representations,
    evaluate_classifier,
    evaluate_language_model,
    find_similar_words,
    preprocess_text,
    represent_texts,
    train_embeddings,
)
from smart_text_intelligence.analysis.compare_similarity import compare_similarity


app = FastAPI(
    title="Smart Text Intelligence API",
    version="1.0.0",
    description="HTTP access to the project's classical NLP functions.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class RepresentRequest(BaseModel):
    documents: list[str] = Field(min_length=1, max_length=100)
    ngram_size: int = Field(default=2, ge=1, le=5)


class DatasetRequest(BaseModel):
    texts: list[str] = Field(min_length=4, max_length=10_000)
    labels: list[str] = Field(min_length=4, max_length=10_000)
    test_size: float = Field(default=0.2, gt=0, lt=0.5)
    random_state: int = Field(default=42, ge=0, le=2_147_483_647)


class ClassifyRequest(DatasetRequest):
    model: Literal["naive_bayes", "logistic_regression"] = "naive_bayes"


class CompareRequest(DatasetRequest):
    include_embeddings: bool = False


class EmbeddingsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=100)
    method: Literal["word2vec", "fasttext", "glove"] = "word2vec"
    vector_size: int = Field(default=50, ge=8, le=300)
    top_n: int = Field(default=8, ge=1, le=30)
    include_projection: bool = False


class LanguageModelRequest(BaseModel):
    training_text: str = Field(min_length=1, max_length=20_000)
    evaluation_text: str = Field(min_length=1, max_length=20_000)
    alpha: float = Field(default=1.0, ge=0, le=10)


def _call_feature(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, TypeError, LookupError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    return _call_feature(analyze_text, request.text)


@app.post("/api/represent")
def represent(request: RepresentRequest):
    return _call_feature(
        represent_texts,
        request.documents,
        ngram_size=request.ngram_size,
    )


@app.post("/api/classify")
def classify(request: ClassifyRequest):
    if len(request.texts) != len(request.labels):
        raise HTTPException(
            status_code=422,
            detail="texts and labels must have the same number of items",
        )
    result = _call_feature(
        evaluate_classifier,
        request.texts,
        request.labels,
        model_name=request.model,
        test_size=request.test_size,
        random_state=request.random_state,
    )
    return {key: value for key, value in result.items() if key != "model"}


@app.post("/api/compare")
def compare(request: CompareRequest):
    if len(request.texts) != len(request.labels):
        raise HTTPException(
            status_code=422,
            detail="texts and labels must have the same number of items",
        )
    return _call_feature(
        compare_dataset_representations,
        request.texts,
        request.labels,
        test_size=request.test_size,
        random_state=request.random_state,
        include_embeddings=request.include_embeddings,
    )


@app.post("/api/embeddings")
def embeddings(request: EmbeddingsRequest):
    if request.include_projection:
        normalized_query = preprocess_text(request.query)["tokens"]
        if not isinstance(normalized_query, list) or len(normalized_query) != 1:
            raise HTTPException(
                status_code=422,
                detail="query must contain exactly one word",
            )
        model = _call_feature(
            train_embeddings,
            request.texts,
            method=request.method,
            vector_size=request.vector_size,
        )
        neighbors = _call_feature(
            compare_similarity,
            model,
            normalized_query[0],
            top_n=request.top_n,
        )
        vectors = getattr(model, "wv", model)
        vocabulary = (
            list(vectors.index_to_key)
            if hasattr(vectors, "index_to_key")
            else list(vectors)
        )
        plotted_words = vocabulary[:40]
        if normalized_query[0] not in plotted_words:
            plotted_words.append(normalized_query[0])
        plotted_words.extend(
            word for word, _ in neighbors if word not in plotted_words
        )
        matrix = np.asarray([vectors[word] for word in plotted_words])
        if len(plotted_words) == 1:
            coordinates = np.zeros((1, 2))
        else:
            coordinates = PCA(n_components=2).fit_transform(matrix)
        return {
            "query": normalized_query[0],
            "neighbors": neighbors,
            "projection": [
                {"word": word, "x": float(point[0]), "y": float(point[1])}
                for word, point in zip(plotted_words, coordinates)
            ],
        }
    return _call_feature(
        find_similar_words,
        request.texts,
        request.query,
        method=request.method,
        top_n=request.top_n,
        vector_size=request.vector_size,
    )


@app.post("/api/language-model")
def language_model(request: LanguageModelRequest):
    result = _call_feature(
        evaluate_language_model,
        request.training_text,
        request.evaluation_text,
        alpha=request.alpha,
    )
    return {key: value for key, value in result.items() if key != "model"}

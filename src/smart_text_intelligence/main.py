import argparse
import json
from collections.abc import Sequence
from typing import Any

from smart_text_intelligence.analysis.extract_keywords import extract_keywords
from smart_text_intelligence.analysis.sentiment_baseline import sentiment_baseline
from smart_text_intelligence.analysis.word_frequency import word_frequency
from smart_text_intelligence.classification.predict_text import predict_text
from smart_text_intelligence.classification.train_logistic_regression import (
    train_logistic_regression,
)
from smart_text_intelligence.classification.train_naive_bayes import (
    train_naive_bayes,
)
from smart_text_intelligence.data.load_dataset import load_labeled_csv
from smart_text_intelligence.data.split_dataset import split_dataset
from smart_text_intelligence.embeddings.train_fasttext import train_fasttext
from smart_text_intelligence.embeddings.train_glove import train_glove
from smart_text_intelligence.embeddings.train_word2vec import train_word2vec
from smart_text_intelligence.analysis.compare_similarity import compare_similarity
from smart_text_intelligence.embeddings.load_glove import load_glove_embeddings
from smart_text_intelligence.embeddings.vector_arithmetic import vector_arithmetic
from smart_text_intelligence.evaluation.classification_metrics import (
    classification_metrics,
)
from smart_text_intelligence.evaluation.error_analysis import error_analysis
from smart_text_intelligence.evaluation.compare_representations import (
    compare_representations,
)
from smart_text_intelligence.language_model.bigram_model import BigramLanguageModel
from smart_text_intelligence.preprocessing.clean_text import clean_text
from smart_text_intelligence.representation.bag_of_words import bag_of_words
from smart_text_intelligence.representation.make_ngrams import make_ngrams
from smart_text_intelligence.representation.one_hot import encode_one_hot
from smart_text_intelligence.representation.tfidf import ManualTfidf
from smart_text_intelligence.preprocessing.expand_contractions import (
    expand_contractions,
)
from smart_text_intelligence.preprocessing.lemmatize_tokens import (
    lemmatize_tokens,
)
from smart_text_intelligence.preprocessing.remove_stop_words import (
    remove_stop_words,
)
from smart_text_intelligence.preprocessing.split_sentences import split_sentences
from smart_text_intelligence.preprocessing.stem_tokens import stem_tokens
from smart_text_intelligence.preprocessing.tokenize_words import tokenize_words


def preprocess_text(text: str) -> dict[str, list[str] | str]:
    """Run the Phase 1 preprocessing features and return each stage's output."""
    expanded = expand_contractions(text)
    cleaned = clean_text(expanded)
    sentences = split_sentences(cleaned)
    tokens = tokenize_words(cleaned)
    filtered_tokens = remove_stop_words(tokens)
    return {
        "expanded_text": expanded,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "without_stop_words": filtered_tokens,
        "stemmed_tokens": stem_tokens(filtered_tokens),
        "lemmatized_tokens": lemmatize_tokens(filtered_tokens),
    }


def analyze_text(text: str) -> dict[str, Any]:
    """Return preprocessing, frequency, keyword, and lexicon-baseline results."""
    processed = preprocess_text(text)
    tokens = processed["without_stop_words"]
    if not isinstance(tokens, list):
        raise TypeError("preprocessing produced an invalid token result")
    return {
        "preprocessing": processed,
        "word_frequency": word_frequency(tokens, top_n=20),
        "keywords": extract_keywords(text, top_n=10),
        "sentiment_baseline": sentiment_baseline(
            tokenize_words(str(processed["expanded_text"]))
        ),
    }


def represent_texts(
    documents: Sequence[str],
    *,
    ngram_size: int = 2,
) -> dict[str, Any]:
    """Run the manual one-hot, BoW, n-gram, and TF-IDF feature modules."""
    tokenized = [
        tokenize_words(clean_text(expand_contractions(document)))
        for document in documents
    ]
    vocabulary, counts = bag_of_words(tokenized)
    tfidf = ManualTfidf(smooth_idf=True, norm="l2")
    tfidf_matrix = tfidf.fit_transform(tokenized) if tokenized else []
    return {
        "vocabulary": vocabulary,
        "one_hot_first_document": (
            encode_one_hot(tokenized[0], vocabulary) if tokenized else {}
        ),
        "bag_of_words": counts,
        "ngrams": [make_ngrams(document, ngram_size) for document in tokenized],
        "tfidf_vocabulary": tfidf.vocabulary_,
        "tfidf": tfidf_matrix,
    }


def train_embeddings(
    texts: Sequence[str],
    *,
    method: str = "word2vec",
    vector_size: int = 50,
    seed: int = 42,
) -> Any:
    """Train one supported static embedding model from raw text examples."""
    sentences = [
        tokenize_words(clean_text(expand_contractions(text)))
        for text in texts
    ]
    if method == "word2vec":
        return train_word2vec(
            sentences,
            vector_size=vector_size,
            seed=seed,
        )
    if method == "fasttext":
        return train_fasttext(
            sentences,
            vector_size=vector_size,
            seed=seed,
        )
    if method == "glove":
        return train_glove(
            sentences,
            vector_size=vector_size,
            seed=seed,
        )
    raise ValueError("method must be 'word2vec', 'fasttext', or 'glove'")


def find_similar_words(
    texts: Sequence[str],
    word: str,
    *,
    method: str = "word2vec",
    top_n: int = 5,
    vector_size: int = 50,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Train an embedding model on supplied text and rank similar known words."""
    model = train_embeddings(
        texts,
        method=method,
        vector_size=vector_size,
        seed=seed,
    )
    return compare_similarity(model, word, top_n=top_n)


def load_pretrained_glove(path: str, *, max_words: int | None = None) -> dict[str, Any]:
    """Load a plain-text GloVe file for similarity or vector arithmetic."""
    return load_glove_embeddings(path, max_words=max_words)


def embedding_analogy(
    model: Any,
    *,
    positive: Sequence[str],
    negative: Sequence[str] = (),
) -> list[float]:
    """Compute a positive-minus-negative embedding vector."""
    return vector_arithmetic(model, positive=positive, negative=negative)


def compare_text_representations(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    train_labels: Sequence[str],
    test_labels: Sequence[str],
    *,
    include_embeddings: bool = True,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run held-out BoW, TF-IDF, and averaged Word2Vec comparisons."""
    return compare_representations(
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        include_embeddings=include_embeddings,
        random_state=random_state,
    )


def compare_dataset_representations(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_embeddings: bool = True,
) -> dict[str, dict[str, Any]]:
    """Split labeled examples once, then compare feature representations."""
    train_texts, test_texts, train_labels, test_labels = split_dataset(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
    )
    return compare_text_representations(
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        include_embeddings=include_embeddings,
        random_state=random_state,
    )


def evaluate_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    model_name: str = "naive_bayes",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split data, fit one text classifier, and report held-out metrics/errors."""
    train_texts, test_texts, train_labels, test_labels = split_dataset(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
    )
    if model_name == "naive_bayes":
        model = train_naive_bayes(train_texts, train_labels)
    elif model_name == "logistic_regression":
        model = train_logistic_regression(
            train_texts,
            train_labels,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "model_name must be 'naive_bayes' or 'logistic_regression'"
        )
    predictions = predict_text(model, test_texts)
    return {
        "model": model,
        "test_texts": test_texts,
        "true_labels": test_labels,
        "predictions": predictions,
        "metrics": classification_metrics(test_labels, predictions),
        "errors": error_analysis(test_texts, test_labels, predictions),
    }


def evaluate_language_model(
    training_text: str,
    evaluation_text: str,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Train a smoothed bigram model and score held-out text."""
    training_sentences = [
        tokenize_words(sentence)
        for sentence in split_sentences(training_text)
    ]
    evaluation_sentences = [
        tokenize_words(sentence)
        for sentence in split_sentences(evaluation_text)
    ]
    model = BigramLanguageModel(alpha=alpha).fit(training_sentences)
    return {
        "model": model,
        "perplexity": model.perplexity(evaluation_sentences),
        "training_sentences": len(training_sentences),
        "evaluation_sentences": len(evaluation_sentences),
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classical NLP features from the Smart Text Intelligence System."
    )
    commands = parser.add_subparsers(dest="command")

    preprocess_parser = commands.add_parser("preprocess", help="preprocess text")
    preprocess_parser.add_argument("text", nargs="?", default="This isn't a bad example.")

    analysis_parser = commands.add_parser("analyze", help="analyze a text passage")
    analysis_parser.add_argument("text")

    representation_parser = commands.add_parser(
        "represent",
        help="create manual numeric representations for documents",
    )
    representation_parser.add_argument("documents", nargs="+")
    representation_parser.add_argument("--ngram-size", type=int, default=2)

    embeddings_parser = commands.add_parser(
        "embeddings",
        help="train a static embedding model and find similar words",
    )
    embeddings_parser.add_argument("word")
    embeddings_parser.add_argument("corpus", nargs="+")
    embeddings_parser.add_argument(
        "--method",
        choices=("word2vec", "fasttext", "glove"),
        default="word2vec",
    )
    embeddings_parser.add_argument("--top-n", type=int, default=5)
    embeddings_parser.add_argument("--vector-size", type=int, default=50)

    classifier_parser = commands.add_parser(
        "classify",
        help="train and evaluate on a labeled CSV dataset",
    )
    classifier_parser.add_argument("csv_path")
    classifier_parser.add_argument("--text-column", default="text")
    classifier_parser.add_argument("--label-column", default="label")
    classifier_parser.add_argument(
        "--model",
        choices=("naive_bayes", "logistic_regression"),
        default="naive_bayes",
    )

    compare_parser = commands.add_parser(
        "compare",
        help="compare BoW, TF-IDF, and optional Word2Vec on labeled CSV data",
    )
    compare_parser.add_argument("csv_path")
    compare_parser.add_argument("--text-column", default="text")
    compare_parser.add_argument("--label-column", default="label")
    compare_parser.add_argument("--without-embeddings", action="store_true")

    language_parser = commands.add_parser(
        "language-model",
        help="train a bigram model and score held-out text",
    )
    language_parser.add_argument("training_text")
    language_parser.add_argument("evaluation_text")
    language_parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "preprocess":
        _print_json(preprocess_text(args.text))
    elif args.command == "analyze":
        _print_json(analyze_text(args.text))
    elif args.command == "represent":
        _print_json(
            represent_texts(
                args.documents,
                ngram_size=args.ngram_size,
            )
        )
    elif args.command == "embeddings":
        _print_json(
            find_similar_words(
                args.corpus,
                args.word,
                method=args.method,
                top_n=args.top_n,
                vector_size=args.vector_size,
            )
        )
    elif args.command == "classify":
        texts, labels = load_labeled_csv(
            args.csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        results = evaluate_classifier(texts, labels, model_name=args.model)
        _print_json({key: value for key, value in results.items() if key != "model"})
    elif args.command == "compare":
        texts, labels = load_labeled_csv(
            args.csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        _print_json(
            compare_dataset_representations(
                texts,
                labels,
                include_embeddings=not args.without_embeddings,
            )
        )
    elif args.command == "language-model":
        results = evaluate_language_model(
            args.training_text,
            args.evaluation_text,
            alpha=args.alpha,
        )
        _print_json({key: value for key, value in results.items() if key != "model"})
    else:
        _print_json(preprocess_text("This isn't a bad example. We're learning NLP."))


if __name__ == "__main__":
    main()

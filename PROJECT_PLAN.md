# Smart Text Intelligence System — Phased Project Plan

## 1. Project goal

Build a modular, tested Python application that demonstrates classical NLP from preprocessing through evaluation. Each feature will live in its own Python file, with matching tests. A small `main.py` will import and call the feature functions to run the selected workflow; it will coordinate features, not contain their implementations.

The project will use classical NLP only. LLMs, generative AI, RAG, and vector databases are out of scope.

## 2. Architecture rules

1. **One feature per source file.** Keep each file focused on one behavior, such as tokenization, stop-word filtering, TF-IDF, or a classifier.
2. **Expose functions from feature files.** Prefer small functions with typed inputs and outputs. Use a class only when the feature genuinely needs persistent state, such as a trained model.
3. **`main.py` is an orchestrator.** It imports feature functions, passes data between them, and selects a workflow. It must not duplicate preprocessing, modeling, or evaluation logic.
4. **Test each feature independently.** Put tests under `tests/`, mirroring the source structure.
5. **Keep reusable code out of notebooks and the UI.** Notebooks are for experiments; Streamlit calls the same tested pipeline used by the command-line entry point.
6. **Make experiments reproducible.** Record dataset sources, split strategy, random seeds, library versions, and model settings.
7. **Compare like with like.** When validating a manual algorithm against a library, document and align their assumptions and configuration.

## 3. Proposed repository structure

```text
smart-text-intelligence/
├── data/
│   ├── raw/                         # Downloaded, unmodified data; do not commit restricted data
│   └── processed/                   # Generated datasets, ignored by version control if appropriate
├── notebooks/                       # Exploration and visualizations only
├── src/
│   └── smart_text_intelligence/
│       ├── __init__.py
│       ├── main.py                  # Thin workflow/CLI entry point
│       ├── config.py                # Paths, seeds, and shared configuration
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load_dataset.py
│       │   └── split_dataset.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── clean_text.py
│       │   ├── split_sentences.py
│       │   ├── tokenize_words.py
│       │   ├── expand_contractions.py
│       │   ├── remove_stop_words.py
│       │   ├── stem_tokens.py
│       │   └── lemmatize_tokens.py
│       ├── representation/
│       │   ├── __init__.py
│       │   ├── one_hot.py
│       │   ├── bag_of_words.py
│       │   ├── make_ngrams.py
│       │   └── tfidf.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── train_word2vec.py
│       │   ├── load_glove.py
│       │   ├── train_fasttext.py
│       │   ├── cosine_similarity.py
│       │   └── vector_arithmetic.py
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── train_naive_bayes.py
│       │   ├── train_logistic_regression.py
│       │   └── predict_text.py
│       ├── language_model/
│       │   ├── __init__.py
│       │   ├── count_ngrams.py
│       │   ├── bigram_probability.py
│       │   ├── smooth_probabilities.py
│       │   └── calculate_perplexity.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── word_frequency.py
│       │   ├── extract_keywords.py
│       │   ├── compare_similarity.py
│       │   └── sentiment_baseline.py
│       └── evaluation/
│           ├── __init__.py
│           ├── classification_metrics.py
│           ├── compare_representations.py
│           └── error_analysis.py
├── app/
│   └── streamlit_app.py             # Added after the core pipeline works
├── tests/
│   ├── preprocessing/
│   ├── representation/
│   ├── embeddings/
│   ├── classification/
│   ├── language_model/
│   ├── analysis/
│   └── evaluation/
├── requirements.txt
├── README.md
└── .gitignore
```

The structure is a starting point. Create a feature file when its phase begins rather than creating empty files for the entire project up front.

## 4. How features connect to `main.py`

Feature implementations remain in their own modules. `main.py` imports and calls them in the required order. For example:

```python
from smart_text_intelligence.preprocessing.clean_text import clean_text
from smart_text_intelligence.preprocessing.tokenize_words import tokenize_words
from smart_text_intelligence.representation.tfidf import build_tfidf_features


def run_preprocessing(text: str) -> list[str]:
    cleaned = clean_text(text)
    return tokenize_words(cleaned)


def main() -> None:
    documents = ["Example text for the pipeline."]
    tokens = [run_preprocessing(document) for document in documents]
    features = build_tfidf_features(tokens)
    print(features)


if __name__ == "__main__":
    main()
```

This is only an architectural example; the functions and their exact signatures will be implemented and tested in their respective phases. Do not copy feature algorithms into `main.py`. For trained models, a feature module can expose a small, consistent set of functions such as `train(...)` and `predict(...)`.

## 5. Phases

### Phase 0 — Project foundation and experiment decisions (code complete; dataset remains configurable)

**Purpose:** Make the project runnable and settle decisions that affect later comparisons.

**Work**
- Create the package and test layout, virtual environment, dependency file, and `.gitignore`.
- Select the first dataset and document its source, license, labels, and permitted use.
- Make the first release task explicit. SMS Spam classification is the suggested starting task.
- Decide which additional datasets are needed for sentiment, embeddings, and language modeling. Do not imply that a spam dataset alone evaluates all those tasks.
- Define data splits, random seeds, baseline methods, and initial evaluation metrics.
- Add a short README with setup and test commands.

**Exit gate**
- A clean environment can install the declared dependencies.
- A documented dataset can be loaded reproducibly.
- The test runner discovers and runs a passing starter test.
- The first-release task and evaluation protocol are written down.

### Phase 1 — Text preprocessing (implemented)

**Purpose:** Turn raw text into consistently cleaned, tokenized text without destroying useful meaning.

**Feature files**
- `clean_text.py`: configurable casing, whitespace, punctuation, and special-character handling.
- `split_sentences.py`: deterministic sentence splitting at terminal punctuation followed by whitespace; abbreviations are a known limitation.
- `tokenize_words.py`: Unicode-aware tokenization that keeps contractions and decimal numbers intact.
- `expand_contractions.py`: expands a defined set of unambiguous English contractions; ambiguous forms remain unchanged.
- `remove_stop_words.py`: NLTK English stop words with negation preserved by default.
- `stem_tokens.py`: NLTK Porter stemming.
- `lemmatize_tokens.py`: NLTK WordNet lemmatization, noun POS by default, with other WordNet POS options.

**Work**
- Explain each operation before implementing it.
- Compare manual/simple behavior with NLTK or spaCy where useful.
- Test empty input, punctuation, contractions, mixed casing, and negation such as “not good.”
- Record how preprocessing choices change tokens and downstream meaning.

**Exit gate**
- Each feature has focused tests and documented inputs/outputs.
- The pipeline does not silently remove negation.
- A short before/after example is documented for every transformation.
- `main.py` composes the feature modules without duplicating their implementation.
- Focused and integration tests pass; the sample pipeline runs successfully.

### Phase 2 — Text representation (implemented)

**Purpose:** Convert tokens/documents into numerical features.

**Feature files**
- `one_hot.py`
- `bag_of_words.py`
- `make_ngrams.py`
- `tfidf.py`

**Work**
- Implement one-hot encoding, Bag of Words, and n-gram counts.
- Implement the selected TF-IDF formula manually.
- Compare with scikit-learn using explicitly documented tokenization, term-frequency, IDF smoothing, and normalization settings.
- Show vocabulary size and sparsity on the chosen dataset.

**Exit gate**
- Tests cover expected vocabulary, counts, unseen terms, and empty documents.
- The manual TF-IDF comparison explains any difference in convention and matches when configurations are aligned.
- Feature matrices can be passed into the classification phase without ad hoc conversion.

### Phase 3 — Classification (implemented)

**Purpose:** Train and evaluate classical text classifiers on labeled text.

**Feature files**
- `train_naive_bayes.py`
- `train_logistic_regression.py`
- `predict_text.py`
- `classification_metrics.py` (created in the evaluation package and reused here)

**Work**
- Specify the Naive Bayes variant and input representation.
- Train Naive Bayes and Logistic Regression using the training split only.
- Compare against a simple baseline.
- Report precision, recall, F1, and a confusion matrix; include accuracy as supplementary context.
- Keep test data out of vectorizer fitting and model training.

**Exit gate**
- Data splitting and model fitting avoid test-data leakage.
- Results can be reproduced from a documented command and seed.
- At least one false positive and false negative are inspected and explained.

### Phase 4 — Statistical language modeling (implemented)

**Purpose:** Estimate word-sequence probabilities and understand sparse-data behavior.

**Feature files**
- `count_ngrams.py`
- `bigram_probability.py`
- `smooth_probabilities.py`
- `calculate_perplexity.py`

**Work**
- Define sentence-boundary tokens, vocabulary construction, and unknown-token handling.
- Implement unigram/bigram counts, maximum-likelihood probabilities, and Laplace smoothing.
- Work through small probability examples by hand.
- Calculate perplexity on held-out text and compare against NLTK where configuration permits.
- Demonstrate zero-frequency and smoothing behavior.

**Exit gate**
- Probability calculations are tested on tiny hand-checkable examples.
- Sentence boundaries and unseen words are handled explicitly.
- Perplexity is computed on held-out data with the convention documented.

### Phase 5 — Word embeddings (implemented)

**Purpose:** Explore dense word vectors, similarity, and out-of-vocabulary behavior.

**Feature files**
- `train_word2vec.py`
- `load_glove.py`
- `train_glove.py`
- `train_fasttext.py`
- `cosine_similarity.py`
- `vector_arithmetic.py`
- `nearest_words()` in `vector_arithmetic.py`

**Work**
- Decide whether each embedding model is trained locally or loaded from a documented source.
- Document corpus size, preprocessing, vector size, window, training settings, and random seed when applicable.
- Compare CBOW and Skip-gram on the same corpus and settings where possible.
- Explain that GloVe availability/training may require a separate source or implementation.
- Test cosine similarity, vector arithmetic, ambiguous words, and unknown-word behavior.
- Contrast static embeddings with contextual embeddings conceptually; contextual-model integration remains out of scope.

**Exit gate**
- Each vector source and its license/provenance are documented.
- Similarity and unknown-word behavior are tested.
- Results are described as dependent on corpus and training configuration, not as universal rankings.

### Phase 6 — Text analysis (implemented)

**Purpose:** Provide practical, interpretable analysis features using the earlier modules.

**Feature files**
- `word_frequency.py`
- `extract_keywords.py`
- `compare_similarity.py`
- `sentiment_baseline.py`

**Work**
- Implement frequency summaries and TF-IDF keyword ranking.
- Reuse the embedding similarity feature where available.
- Define the sentiment approach and dataset/labels before reporting sentiment performance.
- Demonstrate failures caused by negation, ambiguity, and domain-specific language.

**Exit gate**
- Every analysis feature has examples, tests, and stated limitations.
- Sentiment is not presented as validated unless a suitable labeled dataset and evaluation are included.

### Phase 7 — Evaluation and comparison (implemented)

**Purpose:** Compare methods fairly and make strengths and failure cases visible.

**Feature files**
- `compare_representations.py`
- `error_analysis.py`
- Reuse `classification_metrics.py` and the earlier model/representation modules.

**Work**
- Compare BoW, TF-IDF, and embeddings on the same task, data split, and classifier where appropriate.
- Study the effect of selected preprocessing changes using controlled experiments.
- Record metrics, configuration, runtime where useful, and representative errors.
- Explain when a method is a good fit and when it fails.

**Exit gate**
- Results are reproducible and include baselines and suitable metrics.
- Comparisons isolate one meaningful change at a time.
- The results and limitations are summarized in the README.

### Phase 8 — Streamlit and React applications (implemented)

**Purpose:** Provide separate interfaces to the completed, tested pipeline.

**Files**
- `app/streamlit_app.py`
- `api/app.py`: FastAPI adapter that calls the existing Python workflows.
- `frontend/`: independent React + TypeScript application that calls the API.

**Work**
- Let users choose supported tasks and enter or upload text, subject to clear input limits.
- Call the same package functions used by `main.py`; do not duplicate algorithms in UI callbacks.
- Keep React, FastAPI, and Streamlit as distinct layers/apps.
- Display predictions or analysis with explanations of model and known limitations.
- Handle missing models/data and invalid inputs with clear user-facing messages.

**Exit gate**
- Core flows work through the UI and reuse tested feature modules.
- Both interfaces can be started using documented commands.
- React production build and API integration tests pass.
- Errors are explicit and do not produce misleading success results.

### Phase 9 — Portfolio and release readiness (implemented; data-specific evidence awaits dataset selection)

**Purpose:** Make the repository understandable, reproducible, and presentable.

**Work**
- Complete README setup, architecture, algorithms, results, and limitations.
- Include a concise demonstration and representative examples.
- Ensure tests run from a clean setup.
- Remove generated data/model artifacts that should not be committed; document how to obtain them.
- Review dependency licenses and dataset/model provenance.

**Exit gate**
- A new user can set up the project, run tests, reproduce a documented experiment, and launch the app by following the README.
- Every claimed result has a dataset, method, and metric behind it.

## 6. Feature implementation checklist

For every feature file:

1. State the feature's single responsibility and expected input/output.
2. Add or update focused tests.
3. Implement the smallest useful function or model API.
4. Compare with a library when that is part of the feature's learning goal.
5. Document assumptions, limitations, and one worked example.
6. Import and call the feature from `main.py` only when integrating the workflow.
7. Run the focused tests before moving to the next feature.

## 7. Current completion and next decision

All planned code modules are now implemented as separate feature files, with
tests and workflows in `main.py`. The app accepts user-provided text and labeled
CSV data rather than embedding an unreviewed dataset in the repository.

The remaining Phase 0/release decision is to choose and document a dataset and
its license before recording real classification, sentiment, or representation
comparison results. Until then, the repository provides reusable evaluation
code but makes no dataset-specific performance claims.

The Streamlit and React interfaces use the same tested package functions as the
command-line entry point. Start FastAPI and Vite separately to use React; the
existing Streamlit app remains independent. Frontend data stays in memory for
the request and is not persisted.

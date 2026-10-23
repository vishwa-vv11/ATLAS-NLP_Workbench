import io

import streamlit as st

from smart_text_intelligence.data.load_dataset import load_labeled_csv
from smart_text_intelligence.main import (
    analyze_text,
    compare_dataset_representations,
    evaluate_classifier,
    evaluate_language_model,
    find_similar_words,
)


st.set_page_config(page_title="ATLAS — Classical NLP Workbench", page_icon="📝")
st.title("ATLAS — Classical NLP Workbench")
st.caption(
    "Classical NLP learning tools: transparent baselines, no LLMs or generative AI."
)

task = st.sidebar.selectbox(
    "Choose a feature",
    (
        "Text analysis",
        "Text classification",
        "Representation comparison",
        "Word embeddings",
        "Bigram language model",
    ),
)

if task == "Text analysis":
    text = st.text_area(
        "Enter text to analyze",
        "This isn't a bad example. We're learning classical NLP!",
        max_chars=20_000,
        height=180,
    )
    if st.button("Analyze text", type="primary"):
        if not text.strip():
            st.error("Enter some text before running the analysis.")
        else:
            results = analyze_text(text)
            st.subheader("Preprocessing")
            st.json(results["preprocessing"])
            st.subheader("Word frequency")
            st.json(results["word_frequency"])
            st.subheader("TF-IDF keyword baseline")
            st.table(results["keywords"])
            st.subheader("Lexicon sentiment baseline")
            st.json(results["sentiment_baseline"])
            st.caption(
                "The sentiment output is a small educational lexicon baseline, "
                "not a validated sentiment model."
            )

elif task == "Text classification":
    uploaded = st.file_uploader("Upload a labeled CSV", type="csv", max_upload_size=10)
    text_column = st.text_input("Text column", value="text")
    label_column = st.text_input("Label column", value="label")
    model_name = st.selectbox(
        "Classifier",
        ("naive_bayes", "logistic_regression"),
    )
    if uploaded is not None and st.button("Train and evaluate", type="primary"):
        try:
            texts, labels = load_labeled_csv(
                io.StringIO(uploaded.getvalue().decode("utf-8-sig")),
                text_column=text_column,
                label_column=label_column,
            )
            results = evaluate_classifier(
                texts,
                labels,
                model_name=model_name,
            )
            st.subheader("Held-out metrics")
            st.json(results["metrics"])
            st.subheader("Misclassified examples")
            st.dataframe(results["errors"])
            st.caption(
                "Metrics reflect only this uploaded dataset and split; they do "
                "not establish general performance."
            )
        except (UnicodeDecodeError, ValueError, LookupError) as error:
            st.error(str(error))
    elif uploaded is None:
        st.info("Upload a UTF-8 CSV with non-empty text and label columns.")

elif task == "Representation comparison":
    uploaded = st.file_uploader(
        "Upload a labeled CSV",
        type="csv",
        key="compare",
        max_upload_size=10,
    )
    text_column = st.text_input("Text column", value="text", key="compare_text")
    label_column = st.text_input("Label column", value="label", key="compare_label")
    include_embeddings = st.checkbox("Include averaged Word2Vec features", value=True)
    if uploaded is not None and st.button("Compare methods", type="primary"):
        try:
            texts, labels = load_labeled_csv(
                io.StringIO(uploaded.getvalue().decode("utf-8-sig")),
                text_column=text_column,
                label_column=label_column,
            )
            results = compare_dataset_representations(
                texts,
                labels,
                include_embeddings=include_embeddings,
            )
            st.json(results)
            st.caption(
                "BoW/TF-IDF vectorizers and embeddings are fitted on training "
                "text only. Results are dataset-specific."
            )
        except (UnicodeDecodeError, ValueError, LookupError) as error:
            st.error(str(error))
    elif uploaded is None:
        st.info("Upload a labeled CSV to compare representations.")

elif task == "Word embeddings":
    corpus = st.text_area(
        "Training corpus (one or more sentences)",
        "cats purr and sleep. dogs bark and play. cats and dogs are animals.",
        max_chars=20_000,
    )
    method = st.selectbox("Embedding method", ("word2vec", "fasttext", "glove"))
    query = st.text_input("Find words similar to", value="cats")
    if st.button("Train and explore", type="primary"):
        if not corpus.strip() or not query.strip():
            st.error("Both a training corpus and query word are required.")
        else:
            try:
                results = find_similar_words(
                    [corpus],
                    query,
                    method=method,
                    top_n=5,
                    vector_size=50,
                )
                st.json(results)
                st.caption(
                    "Embeddings trained on small corpora are educational "
                    "demonstrations; similarity quality is corpus-dependent."
                )
            except KeyError as error:
                st.error(str(error))

else:
    st.write(
        "Enter separate training and evaluation passages. Perplexity is "
        "meaningful only when the evaluation text is held out."
    )
    training_text = st.text_area(
        "Training text",
        "classical language models learn word patterns. language models "
        "estimate the next word.",
        max_chars=20_000,
    )
    evaluation_text = st.text_area(
        "Evaluation text",
        "language models estimate word patterns.",
        max_chars=20_000,
    )
    alpha = st.number_input("Laplace smoothing alpha", min_value=0.0, value=1.0)
    if st.button("Calculate perplexity", type="primary"):
        if not training_text.strip() or not evaluation_text.strip():
            st.error("Both training and evaluation text are required.")
        else:
            results = evaluate_language_model(
                training_text,
                evaluation_text,
                alpha=alpha,
            )
            st.metric("Held-out perplexity", f"{results['perplexity']:.3f}")
            st.caption(
                "Lower perplexity indicates better fit to this tokenized "
                "evaluation sample; it does not measure general text quality."
            )

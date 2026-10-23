from sklearn.feature_extraction.text import TfidfVectorizer

from smart_text_intelligence.preprocessing.clean_text import clean_text
from smart_text_intelligence.preprocessing.expand_contractions import (
    expand_contractions,
)
from smart_text_intelligence.preprocessing.remove_stop_words import remove_stop_words
from smart_text_intelligence.preprocessing.tokenize_words import tokenize_words


def extract_keywords(
    text: str,
    *,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Rank single-word terms in one document using TF-IDF as a simple baseline."""
    if top_n < 0:
        raise ValueError("top_n must not be negative")
    if not text.strip() or top_n == 0:
        return []
    normalized = clean_text(expand_contractions(text))
    tokens = remove_stop_words(tokenize_words(normalized))
    if not tokens:
        return []
    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
    )
    matrix = vectorizer.fit_transform([" ".join(tokens)])
    names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]
    ranked = sorted(
        ((str(name), float(score)) for name, score in zip(names, scores)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked[:top_n]

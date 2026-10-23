from collections.abc import Sequence

from nltk.stem import PorterStemmer


def stem_tokens(tokens: Sequence[str]) -> list[str]:
    """Stem tokens with NLTK's Porter stemming algorithm."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

from collections.abc import Collection, Sequence

from nltk.corpus import stopwords


_NEGATION_TERMS = {"cannot", "never", "no", "not", "nor", "n't"}


def remove_stop_words(
    tokens: Sequence[str],
    *,
    stop_words: Collection[str] | None = None,
    preserve_negation: bool = True,
) -> list[str]:
    """Remove English stop words, preserving negation terms by default.

    If ``stop_words`` is omitted, the NLTK ``stopwords`` corpus is used.
    Install it with ``python -m nltk.downloader stopwords``.
    """
    if stop_words is None:
        try:
            stop_words = stopwords.words("english")
        except LookupError as error:
            raise LookupError(
                "NLTK stopwords data is missing. Run "
                "'python -m nltk.downloader stopwords'."
            ) from error

    normalized_stop_words = {word.lower() for word in stop_words}
    if preserve_negation:
        normalized_stop_words.difference_update(_NEGATION_TERMS)

    return [
        token
        for token in tokens
        if token.lower() not in normalized_stop_words
    ]

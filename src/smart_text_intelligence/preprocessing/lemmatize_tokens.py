from collections.abc import Sequence
from typing import Literal

from nltk.stem import WordNetLemmatizer


WordNetPartOfSpeech = Literal["n", "v", "a", "r", "s"]


def lemmatize_tokens(
    tokens: Sequence[str],
    *,
    part_of_speech: WordNetPartOfSpeech = "n",
) -> list[str]:
    """Lemmatize tokens with NLTK WordNet, using noun POS by default.

    Install the required NLTK data with
    ``python -m nltk.downloader wordnet omw-1.4``.
    """
    if part_of_speech not in {"n", "v", "a", "r", "s"}:
        raise ValueError("part_of_speech must be a valid WordNet POS: n, v, a, r, or s")

    lemmatizer = WordNetLemmatizer()
    try:
        return [
            lemmatizer.lemmatize(token, pos=part_of_speech)
            for token in tokens
        ]
    except LookupError as error:
        raise LookupError(
            "NLTK WordNet data is missing. Run "
            "'python -m nltk.downloader wordnet omw-1.4'."
        ) from error

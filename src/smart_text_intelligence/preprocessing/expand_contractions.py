import re


_CONTRACTIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'll": "he will",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'll": "she will",
    "shouldn't": "should not",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_CONTRACTION_PATTERN = re.compile(
    r"\b[^\W_]+(?:['\u2019][^\W_]+)+\b",
    re.UNICODE,
)


def _match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def expand_contractions(text: str) -> str:
    """Expand supported English contractions while preserving surrounding text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        normalized = original.replace("\u2019", "'").lower()
        replacement = _CONTRACTIONS.get(normalized)
        return _match_case(replacement, original) if replacement else original

    return _CONTRACTION_PATTERN.sub(replace, text)

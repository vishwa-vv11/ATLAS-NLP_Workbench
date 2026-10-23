from collections.abc import Iterable


DEFAULT_POSITIVE_WORDS = {
    "amazing", "awesome", "excellent", "good", "great", "happy",
    "love", "loved", "nice", "wonderful",
}
DEFAULT_NEGATIVE_WORDS = {
    "awful", "bad", "boring", "hate", "hated", "horrible",
    "poor", "sad", "terrible", "worst",
}
_NEGATIONS = {"never", "no", "not", "n't"}


def sentiment_baseline(
    tokens: Iterable[str],
    *,
    positive_words: set[str] | None = None,
    negative_words: set[str] | None = None,
) -> dict[str, int | str]:
    """Score a tiny lexicon baseline; this is not a validated sentiment model."""
    positive = {word.lower() for word in (positive_words or DEFAULT_POSITIVE_WORDS)}
    negative = {word.lower() for word in (negative_words or DEFAULT_NEGATIVE_WORDS)}
    words = [token.lower() for token in tokens]
    score = 0
    negated = False
    for word in words:
        if word in _NEGATIONS:
            negated = True
            continue
        contribution = int(word in positive) - int(word in negative)
        if contribution:
            score += -contribution if negated else contribution
            negated = False
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {"label": label, "score": score}

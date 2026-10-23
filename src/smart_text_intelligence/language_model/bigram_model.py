from collections import Counter
from collections.abc import Sequence

from smart_text_intelligence.language_model.bigram_probability import (
    bigram_probability,
)
from smart_text_intelligence.language_model.calculate_perplexity import (
    calculate_perplexity,
)
from smart_text_intelligence.language_model.count_ngrams import count_ngrams


class BigramLanguageModel:
    """Train and query a boundary-aware add-alpha bigram language model."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must not be negative")
        self.alpha = alpha
        self.counts: Counter[tuple[str, str]] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, sentences: Sequence[Sequence[str]]) -> "BigramLanguageModel":
        if not sentences:
            raise ValueError("at least one sentence is required")
        self.vocabulary = {token for sentence in sentences for token in sentence}
        self.vocabulary.update({"</s>", "<UNK>"})
        mapped = [
            [token if token in self.vocabulary else "<UNK>" for token in sentence]
            for sentence in sentences
        ]
        self.counts = count_ngrams(mapped, n=2)
        return self

    def probability(self, previous_token: str, token: str) -> float:
        if not self.counts:
            raise ValueError("fit must be called before probability")
        previous = (
            previous_token
            if previous_token == "<s>" or previous_token in self.vocabulary
            else "<UNK>"
        )
        current = token if token in self.vocabulary else "<UNK>"
        return bigram_probability(
            self.counts,
            previous,
            current,
            vocabulary_size=len(self.vocabulary),
            alpha=self.alpha,
        )

    def perplexity(self, sentences: Sequence[Sequence[str]]) -> float:
        if not self.counts:
            raise ValueError("fit must be called before perplexity")
        return calculate_perplexity(
            sentences,
            self.counts,
            vocabulary=self.vocabulary,
            alpha=self.alpha,
        )

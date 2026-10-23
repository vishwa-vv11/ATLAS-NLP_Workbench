import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class ManualTfidf:
    """Small dense TF-IDF transformer for learning and reproducible examples.

    With ``smooth_idf=False``, IDF is log(N / df). With smoothing enabled,
    IDF matches scikit-learn's log((1 + N) / (1 + df)) + 1 convention.
    """

    smooth_idf: bool = False
    norm: str | None = None
    vocabulary_: list[str] = field(default_factory=list, init=False)
    idf_: dict[str, float] = field(default_factory=dict, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def fit(self, documents: Sequence[Sequence[str]]) -> "ManualTfidf":
        if not documents:
            raise ValueError("at least one document is required")
        if self.norm not in {None, "l1", "l2"}:
            raise ValueError("norm must be None, 'l1', or 'l2'")
        self.vocabulary_ = list(
            dict.fromkeys(token for document in documents for token in document)
        )
        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        document_count = len(documents)
        if self.smooth_idf:
            self.idf_ = {
                token: math.log((1 + document_count) / (1 + document_frequency[token]))
                + 1
                for token in self.vocabulary_
            }
        else:
            self.idf_ = {
                token: math.log(document_count / document_frequency[token])
                for token in self.vocabulary_
            }
        self._is_fitted = True
        return self

    def transform(self, documents: Sequence[Sequence[str]]) -> list[list[float]]:
        if not self._is_fitted:
            raise ValueError("fit must be called before transform")
        rows: list[list[float]] = []
        for document in documents:
            counts = Counter(document)
            row = [
                counts[token] * self.idf_.get(token, 0.0)
                for token in self.vocabulary_
            ]
            if self.norm == "l2":
                length = math.sqrt(sum(value * value for value in row))
                if length:
                    row = [value / length for value in row]
            elif self.norm == "l1":
                length = sum(abs(value) for value in row)
                if length:
                    row = [value / length for value in row]
            rows.append(row)
        return rows

    def fit_transform(
        self,
        documents: Sequence[Sequence[str]],
    ) -> list[list[float]]:
        return self.fit(documents).transform(documents)

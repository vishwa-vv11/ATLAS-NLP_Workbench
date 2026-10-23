from io import StringIO

import pytest

from smart_text_intelligence.data.load_dataset import load_labeled_csv
from smart_text_intelligence.data.split_dataset import split_dataset


def test_load_labeled_csv_reads_custom_columns() -> None:
    texts, labels = load_labeled_csv(
        StringIO("message,category\nhello,positive\nbye,negative\n"),
        text_column="message",
        label_column="category",
    )
    assert texts == ["hello", "bye"]
    assert labels == ["positive", "negative"]


def test_load_labeled_csv_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="row 2"):
        load_labeled_csv(StringIO("text,label\n,positive\n"))


def test_split_dataset_is_reproducible_and_stratified() -> None:
    texts = [f"sample {index}" for index in range(20)]
    labels = ["ham"] * 10 + ["spam"] * 10

    first = split_dataset(texts, labels, random_state=7)
    second = split_dataset(texts, labels, random_state=7)
    assert first == second
    assert set(first[2]) == set(first[3]) == {"ham", "spam"}


def test_split_dataset_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        split_dataset(["one"], ["a", "b"])

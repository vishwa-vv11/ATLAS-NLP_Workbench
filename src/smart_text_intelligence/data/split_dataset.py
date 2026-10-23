from collections import Counter

from sklearn.model_selection import train_test_split


def split_dataset(
    texts: list[str],
    labels: list[str],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split labeled examples reproducibly, stratifying when every class permits."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same number of items")
    if len(texts) < 2:
        raise ValueError("at least two labeled examples are required")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    label_counts = Counter(labels)
    can_stratify = (
        stratify
        and len(label_counts) > 1
        and min(label_counts.values()) >= 2
        and round(len(texts) * test_size) >= len(label_counts)
        and round(len(texts) * (1 - test_size)) >= len(label_counts)
    )
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if can_stratify else None,
    )
    return train_texts, test_texts, train_labels, test_labels

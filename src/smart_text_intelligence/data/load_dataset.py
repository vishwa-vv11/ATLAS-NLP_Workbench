import csv
from pathlib import Path
from typing import TextIO


def load_labeled_csv(
    path: str | Path | TextIO,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[str]]:
    """Load non-empty text and labels from a UTF-8 CSV file."""
    texts: list[str] = []
    labels: list[str] = []
    if hasattr(path, "read"):
        file = path
        should_close = False
    else:
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")
        file = csv_path.open("r", encoding="utf-8-sig", newline="")
        should_close = True
    try:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV must include a header row")
        missing_columns = {text_column, label_column} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Dataset CSV is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text or not label:
                raise ValueError(
                    f"Dataset row {row_number} must have non-empty text and label"
                )
            texts.append(text)
            labels.append(label)
    finally:
        if should_close:
            file.close()

    if not texts:
        raise ValueError("Dataset CSV contains no data rows")
    return texts, labels

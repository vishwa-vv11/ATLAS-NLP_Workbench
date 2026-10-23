import re


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    preserve_sentence_punctuation: bool = True,
    preserve_apostrophes: bool = True,
) -> str:
    """Normalize whitespace and remove unsupported characters from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned = text.lower() if lowercase else text
    allowed_punctuation = ".,!?" if preserve_sentence_punctuation else ""
    allowed_apostrophes = "'" + chr(0x2019) if preserve_apostrophes else ""
    allowed = re.escape(allowed_punctuation + allowed_apostrophes)
    cleaned = re.sub(rf"[^\w\s{allowed}]", " ", cleaned, flags=re.UNICODE)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

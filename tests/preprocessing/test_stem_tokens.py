from smart_text_intelligence.preprocessing.stem_tokens import stem_tokens


def test_stem_tokens_applies_porter_stemming() -> None:
    assert stem_tokens(["running", "studies", "ponies"]) == [
        "run",
        "studi",
        "poni",
    ]

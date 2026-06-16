import pytest

from pdf_utils import chunk_text


def test_chunk_text_splits_long_text_with_overlap():
    chunks = chunk_text("abcdefghij", max_chars=5, overlap=2)

    assert chunks == ["abcde", "defgh", "ghij"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", max_chars=5, overlap=5)

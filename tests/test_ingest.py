"""Tests for the chunking logic."""

import pytest

from src.ingest import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("python sql aws", chunk_size=256, overlap=40) == ["python sql aws"]


def test_long_text_splits_into_multiple_chunks():
    words = " ".join(str(i) for i in range(600))
    chunks = chunk_text(words, chunk_size=256, overlap=40)
    assert len(chunks) > 1
    # First chunk holds chunk_size words.
    assert len(chunks[0].split()) == 256


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(300))
    chunks = chunk_text(words, chunk_size=256, overlap=40)
    tail = chunks[0].split()[-40:]
    head = chunks[1].split()[:40]
    assert tail == head  # the overlap region is shared


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)

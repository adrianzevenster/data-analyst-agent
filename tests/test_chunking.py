from __future__ import annotations

import pytest
from app.ingestion.chunking import chunk_text


def test_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_single_chunk():
    text = "Hello world. This is a test."
    chunks = chunk_text(text, max_chars=1200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraph_boundaries_respected():
    # Two clear paragraphs that together exceed max_chars
    para1 = "A" * 600
    para2 = "B" * 600
    text = para1 + "\n\n" + para2
    chunks = chunk_text(text, max_chars=700, overlap=0)
    # para1 and para2 should land in separate chunks
    assert any(para1 in c for c in chunks)
    assert any(para2 in c for c in chunks)


def test_paragraphs_merged_when_small():
    # Two short paragraphs should be merged into one chunk
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    assert "Short paragraph one." in chunks[0]
    assert "Short paragraph two." in chunks[0]


def test_sentence_split_for_oversized_paragraph():
    # Build a paragraph that exceeds max_chars but can be split at sentence boundary
    sent = "This is a sentence. "
    para = sent * 80  # ~1600 chars
    chunks = chunk_text(para, max_chars=500, overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 550  # allow a tiny amount of slack for word-boundary effects


def test_overlap_prefix_appears_in_next_chunk():
    para1 = "First paragraph content here. " * 30   # ~900 chars
    para2 = "Second paragraph content here. " * 30  # ~930 chars
    text = para1 + "\n\n" + para2
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    # The tail of chunk 0 should appear at the start of chunk 1
    tail = chunks[0][-100:].strip()
    assert tail in chunks[1]


def test_no_overlap_flag():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, max_chars=15, overlap=0)
    assert all(len(c) <= 20 for c in chunks)  # no excess from overlap


def test_hard_character_fallback_for_no_sentences():
    # A single very long word (no sentence boundaries)
    text = "x" * 3000
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) >= 5
    for c in chunks:
        assert len(c) <= 560  # max_chars + overlap headroom


def test_single_newlines_not_treated_as_paragraph_break():
    # Single newlines should be treated as inline whitespace, not paragraph breaks
    text = "Line one.\nLine two.\nLine three."
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1


def test_multiple_blank_lines_treated_as_one_paragraph_break():
    para1 = "First."
    para2 = "Second."
    text = para1 + "\n\n\n\n" + para2
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) == 1
    assert "First." in chunks[0]
    assert "Second." in chunks[0]

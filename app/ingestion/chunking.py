from __future__ import annotations

import re

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    sentences = _SENTENCE_END.split(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into chunks that respect paragraph and sentence boundaries.

    Strategy:
    1. Split on paragraph breaks (2+ newlines).
    2. Merge short paragraphs into a chunk until we'd exceed max_chars.
    3. Paragraphs exceeding max_chars are split at sentence boundaries first,
       then character-split as a last resort.
    4. Carry the last `overlap` characters of the previous chunk into the next
       chunk so retrieval doesn't miss meaning at boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    # ── 1. Paragraph splitting ────────────────────────────────────────────────
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    # ── 2. Expand oversized paragraphs at sentence boundaries ─────────────────
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(_split_by_sentences(para, max_chars))

    # ── 3. Merge small units until adding the next would exceed max_chars ─────
    raw_chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for unit in units:
        sep_len = 2 if current_parts else 0  # "\n\n"
        if current_len + sep_len + len(unit) <= max_chars:
            current_parts.append(unit)
            current_len += sep_len + len(unit)
        else:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts))
            if len(unit) <= max_chars:
                current_parts = [unit]
                current_len = len(unit)
            else:
                # Unit still too large — hard character-split with overlap
                i = 0
                while i < len(unit):
                    raw_chunks.append(unit[i:i + max_chars].strip())
                    i += max(1, max_chars - overlap)
                current_parts = []
                current_len = 0

    if current_parts:
        raw_chunks.append("\n\n".join(current_parts))

    if not raw_chunks:
        return []

    # ── 4. Inject overlap prefix from previous chunk ───────────────────────────
    if overlap <= 0:
        return [c for c in raw_chunks if c]

    result: list[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:].strip()
        chunk = raw_chunks[i]
        if prev_tail and not chunk.startswith(prev_tail):
            chunk = prev_tail + "\n\n" + chunk
        result.append(chunk.strip())

    return [c for c in result if c]

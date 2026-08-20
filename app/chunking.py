"""Разбивка текста документа на чанки для индексации."""
from __future__ import annotations

import re


def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Режет текст на перекрывающиеся куски по границам предложений.

    Перекрытие нужно, чтобы факт, стоящий на стыке двух чанков, не терялся
    целиком в одном из них.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) > max_chars:
            for i in range(0, len(sentence), max_chars - overlap):
                chunks.append(sentence[i : i + max_chars])
            current = ""
        else:
            current = sentence

    if current:
        chunks.append(current)

    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail} {cur}".strip())
        return overlapped

    return chunks

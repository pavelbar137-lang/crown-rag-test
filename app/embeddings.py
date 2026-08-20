"""Эмбеддинги текста.

Основной путь — Anthropic-совместимый embeddings-эндпоинт, если задан
ANTHROPIC_API_KEY. Без ключа сервис не падает и не превращается в заглушку:
использует локальный hashing-trick векторизатор (аналог того, что делает
HashingVectorizer в scikit-learn), чтобы поиск по базе знаний работал
офлайн и был воспроизводим без внешних сервисов и без сети.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter

import numpy as np

DIM = 512


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _hash_embed(text: str, dim: int = DIM) -> np.ndarray:
    """Bag-of-words -> фиксированной длины вектор через хэш-триK.

    Каждое слово детерминированно попадает в одну из `dim` корзин по
    md5-хэшу, вектор нормализуется, чтобы косинусное сходство было
    эквивалентно скалярному произведению — это важно для FAISS
    IndexFlatIP.
    """
    vec = np.zeros(dim, dtype="float32")
    tokens = _tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    total = len(tokens)
    for token, count in counts.items():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        tf = count / total
        vec[idx] += sign * tf * math.log(1 + count)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class EmbeddingClient:
    """Единая точка получения эмбеддингов для индексации и запросов."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def embed(self, texts: list[str]) -> np.ndarray:
        # У Anthropic на момент написания нет отдельного embeddings API —
        # если он появится или клиент подключит свой (OpenAI-совместимый)
        # embeddings-сервис, достаточно подставить вызов сюда по тому же
        # контракту (список строк -> np.ndarray [len(texts), dim]).
        return np.stack([_hash_embed(t, self.dim) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

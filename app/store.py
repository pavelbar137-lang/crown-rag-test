"""Векторное хранилище поверх FAISS."""
from __future__ import annotations

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[str] = []

    def add(self, vectors: np.ndarray, chunks: list[str]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors и chunks должны быть одной длины")
        if len(vectors) == 0:
            return
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

"""Мини-сервис RAG: загрузка документа, поиск по базе знаний, ответ на вопрос."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.chunking import split_into_chunks
from app.embeddings import EmbeddingClient
from app.llm import AnswerGenerator
from app.store import VectorStore

app = FastAPI(title="RAG-сервис по базе знаний")

embedder = EmbeddingClient()
store = VectorStore(dim=embedder.dim)
generator = AnswerGenerator()


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "Ожидается текстовый файл в UTF-8")

    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(400, "Документ пуст после нормализации")

    vectors = embedder.embed(chunks)
    store.add(vectors, chunks)
    return IngestResponse(chunks_indexed=len(chunks), total_chunks=len(store.chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(400, "Вопрос пуст")

    query_vector = embedder.embed_one(request.question)
    hits = store.search(query_vector, top_k=request.top_k)
    contexts = [chunk for chunk, _score in hits]

    answer = generator.answer(request.question, contexts)
    return AskResponse(answer=answer, sources=contexts, mode=generator.mode)


@app.delete("/reset")
async def reset() -> dict:
    store.reset()
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(store.chunks), "llm_mode": generator.mode}

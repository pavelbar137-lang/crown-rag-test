from app.chunking import split_into_chunks
from app.embeddings import EmbeddingClient
from app.store import VectorStore


DOCUMENT = (
    "Регламент компании. Раздел 1: Отпуск оформляется через кабинет "
    "сотрудника не позднее чем за 14 дней. Раздел 2: Возврат оборудования "
    "при увольнении обязателен в течение 3 рабочих дней. Раздел 3: "
    "Компенсация мобильной связи выплачивается один раз в квартал."
)


def test_split_into_chunks_covers_whole_document():
    chunks = split_into_chunks(DOCUMENT, max_chars=60, overlap=10)
    assert chunks
    assert all(len(c) <= 80 for c in chunks)
    joined = " ".join(chunks)
    assert "Отпуск" in joined and "Компенсация" in joined


def test_split_into_chunks_empty_text():
    assert split_into_chunks("   ") == []


def test_vector_store_retrieves_relevant_chunk():
    embedder = EmbeddingClient(dim=128)
    chunks = split_into_chunks(DOCUMENT, max_chars=60, overlap=10)
    store = VectorStore(dim=embedder.dim)
    store.add(embedder.embed(chunks), chunks)

    query = embedder.embed_one("сколько дней на возврат оборудования при увольнении")
    hits = store.search(query, top_k=1)

    assert hits
    top_chunk, score = hits[0]
    assert "оборудования" in top_chunk
    assert score > 0


def test_vector_store_empty_search_returns_nothing():
    store = VectorStore(dim=8)
    embedder = EmbeddingClient(dim=8)
    result = store.search(embedder.embed_one("вопрос"), top_k=3)
    assert result == []

from dataclasses import dataclass

from app.core.config import settings
from app.embeddings.base import Embedder, resolve_embedder
from app.vectorstore.pinecone_store import PineconeVectorStore, rerank


@dataclass
class RetrievedChunk:

    vector_id: str
    text: str
    document_id: int
    chunk_index: int
    score: float | None
    filename: str | None = None
    page: int | None = None


class Retriever:
    def __init__(
        self, store: PineconeVectorStore | None = None, embedder: Embedder | None = None
    ) -> None:
        self.store = store or PineconeVectorStore()
        self.embedder = embedder or resolve_embedder()

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        dense = (await self.embedder.embed([query]))[0]
        hits = await self.store.hybrid_query(
            query, dense, top_k=settings.retrieval_candidates
        )
        texts = [(hit.get("metadata") or {}).get("text", "") for hit in hits]
        ranked = await rerank(query, texts, top_k)

        chunks: list[RetrievedChunk] = []
        for original_index, score in ranked:
            hit = hits[original_index]
            metadata = hit.get("metadata") or {}
            page_raw = metadata.get("page")
            try:
                page = int(page_raw) if page_raw is not None else None
            except (TypeError, ValueError):
                page = None
            chunks.append(
                RetrievedChunk(
                    vector_id=hit.get("id", ""),
                    text=metadata.get("text", ""),
                    document_id=int(metadata.get("document_id", 0)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=score,
                    filename=metadata.get("filename"),
                    page=page,
                )
            )
        return chunks
import asyncio
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings

_pc: Pinecone | None = None
_index: Any = None

SPARSE_MODEL = "pinecone-sparse-english-v0"
RERANK_MODEL = "bge-reranker-v2-m3"


def get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        assert settings.pinecone_api_key, "PINECONE_API_KEY is not set in .env"
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index() -> Any:
    global _index
    if _index is None:
        _index = get_pc().Index(settings.pinecone_index_name)
    return _index


def ensure_index(dimension: int) -> None:
    pc = get_pc()
    existing = {item.name: item for item in pc.list_indexes()}
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
        )
        return
    actual = existing[settings.pinecone_index_name].dimension
    if actual != dimension:
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' is {actual} dims but the "
            f"active embedder produces {dimension}. Recreate the index or switch "
            "EMBEDDER_PROVIDER (local/JINA_API_KEY) before re-ingesting."
        )


async def sparse_vectors(
    texts: list[str], input_type: str = "query"
) -> list[dict[str, list[int] | list[float]]]:
    batch = settings.embed_batch_size
    vectors: list[dict[str, list[int] | list[float]]] = []
    for start in range(0, len(texts), batch):
        result = await asyncio.to_thread(
            get_pc().inference.embed,
            model=SPARSE_MODEL,
            inputs=texts[start : start + batch],
            parameters={"input_type": input_type},
        )
        vectors.extend(
            {"indices": item.sparse_indices, "values": item.sparse_values}
            for item in result.data
        )
    return vectors


async def rerank(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    result = await asyncio.to_thread(
        get_pc().inference.rerank,
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )
    return [(item.index, item.score) for item in result.data]


class PineconeVectorStore:
    async def upsert(
        self,
        vector_id: str,
        dense: list[float],
        sparse: dict[str, Any],
        metadata: dict[str, str],
    ) -> None:
        index = get_index()
        await asyncio.to_thread(
            index.upsert,
            vectors=[
                {
                    "id": vector_id,
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": metadata,
                }
            ],
        )

    async def hybrid_query(
        self, query_text: str, dense: list[float], top_k: int
    ) -> list[dict]:
        index = get_index()
        sparse = (await sparse_vectors([query_text], input_type="query"))[0]
        result = await asyncio.to_thread(
            index.query,
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            include_metadata=True,
        )
        return [
            {"id": hit.id, "metadata": hit.metadata, "score": hit.score}
            for hit in result.matches
        ]

    async def delete_by_document(self, document_id: int) -> None:
        index = get_index()
        await asyncio.to_thread(index.delete, filter={"document_id": str(document_id)})

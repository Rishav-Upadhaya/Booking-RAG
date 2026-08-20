from typing import Any, Protocol


class VectorStore(Protocol):
    async def upsert(
        self,
        vector_id: str,
        dense: list[float],
        sparse: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None: ...

    async def hybrid_query(
        self,
        query_text: str,
        dense: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]: ...

    async def delete_by_document(self, document_id: int) -> None: ...

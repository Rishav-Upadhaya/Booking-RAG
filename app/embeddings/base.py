from typing import Protocol

from app.core.config import settings


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


def resolve_embedder() -> Embedder:
    from app.embeddings.jina_embedder import JinaEmbedder
    from app.embeddings.sentence_embedder import SentenceEmbedder

    if settings.embedder_provider == "jina" or (
        settings.embedder_provider == "auto" and settings.jina_api_key
    ):
        return JinaEmbedder()
    return SentenceEmbedder()

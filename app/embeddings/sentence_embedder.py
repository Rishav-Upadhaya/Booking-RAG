import asyncio

from sentence_transformers import SentenceTransformer

from app.core.config import settings

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedder_model)


class SentenceEmbedder:
    def __init__(self) -> None:
        load_model()

    @property
    def dimension(self) -> int:
        return 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        assert _model is not None
        return await asyncio.to_thread(
            lambda: _model.encode(texts, normalize_embeddings=True).tolist()
        )

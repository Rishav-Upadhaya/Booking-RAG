from openai import AsyncOpenAI

from app.core.config import settings


class JinaEmbedder:
    def __init__(self) -> None:
        if not settings.jina_api_key:
            raise RuntimeError("EMBEDDER_PROVIDER=jina requires JINA_API_KEY in .env")
        self._client = AsyncOpenAI(
            base_url="https://api.jina.ai/v1", api_key=settings.jina_api_key
        )

    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, texts: list[str]) -> list[list[float]]:
        batch = settings.embed_batch_size
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch):
            response = await self._client.embeddings.create(
                model=settings.jina_model,
                input=texts[start : start + batch],
                encoding_format="float",
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

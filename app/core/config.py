from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "palm-rag"
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    redis_url: str = "redis://localhost:6380/0"

    pinecone_api_key: str = ""
    pinecone_index_name: str = "documents"
    pinecone_region: str = "us-east-1"

    embedder_provider: str = "auto"
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v3"
    embedder_model: str = "all-MiniLM-L6-v2"

    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    chunk_size: int = 500
    chunk_overlap: int = 50
    max_upload_bytes: int = 50 * 1024 * 1024
    embed_batch_size: int = 64
    max_memory_turns: int = 20
    memory_ttl_seconds: int = 86400
    intent_domain_threshold: float = 0.15
    top_k: int = 5
    retrieval_candidates: int = 20

    @field_validator(
        "pinecone_api_key", "jina_api_key", "openrouter_api_key", mode="before"
    )
    @classmethod
    def strip_key_whitespace(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


settings = Settings()

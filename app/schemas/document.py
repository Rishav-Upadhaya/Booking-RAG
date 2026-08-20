from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.chunking.base import ChunkingStrategy


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DocumentOut(ORMModel):
    id: int
    filename: str
    content_type: str
    status: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    error: str | None = None
    created_at: datetime


class DocumentUploadOut(BaseModel):
    id: int
    status: str = "pending"
    message: str = "upload accepted, processing in background"


class ChunkOut(ORMModel):
    id: int
    document_id: int
    chunk_index: int
    text: str
    vector_id: str
    filename: str | None = None
    page: int | None = None

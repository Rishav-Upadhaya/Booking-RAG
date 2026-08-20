from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.core.config import settings
from app.extraction.base import PageSegment


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


@dataclass
class ChunkPiece:
    text: str
    page: int | None


class Chunker(Protocol):
    async def chunk(self, segments: list[PageSegment]) -> list[ChunkPiece]: ...


def get_chunker(strategy: ChunkingStrategy) -> Chunker:
    from app.chunking.fixed_size import FixedSizeChunker
    from app.chunking.recursive import RecursiveChunker

    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        )
    if strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")

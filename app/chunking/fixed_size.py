import re

from app.chunking.base import ChunkPiece
from app.extraction.base import PageSegment

_TOKEN_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


class FixedSizeChunker:

    def __init__(self, size: int = 500, overlap: int = 50) -> None:
        self.size = size
        self.overlap = overlap

    async def chunk(self, segments: list[PageSegment]) -> list[ChunkPiece]:
        pieces: list[ChunkPiece] = []
        for segment in segments:
            for text in self._chunk_text(segment.text):
                pieces.append(ChunkPiece(text=text, page=segment.number))
        return pieces

    def _chunk_text(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return []
        step = max(self.size - self.overlap, 1)
        chunks = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.size]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + self.size >= len(tokens):
                break
        return chunks

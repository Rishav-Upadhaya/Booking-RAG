import re

from app.chunking.base import ChunkPiece
from app.chunking.fixed_size import count_tokens
from app.extraction.base import PageSegment

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class RecursiveChunker:
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
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        buffer = ""
        for para in paragraphs:
            if count_tokens(para) > self.size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split_long(para))
                continue
            if buffer and count_tokens(buffer) + count_tokens(para) > self.size:
                chunks.append(buffer)
                buffer = ""
            buffer = f"{buffer}\n\n{para}".strip()
        if buffer:
            chunks.append(buffer)
        return self._reflow(chunks)

    def _split_long(self, para: str) -> list[str]:
        pieces = _SENTENCE_RE.split(para)
        result: list[str] = []
        buffer = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if count_tokens(piece) > self.size:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                words = piece.split()
                for start in range(0, len(words), self.size):
                    result.append(" ".join(words[start : start + self.size]))
                continue
            if buffer and count_tokens(buffer) + count_tokens(piece) > self.size:
                result.append(buffer)
                buffer = ""
            buffer = f"{buffer} {piece}".strip()
        if buffer:
            result.append(buffer)
        return result

    def _reflow(self, chunks: list[str]) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            prefix = self._overlap_tail(out[-1]) if out else ""
            if prefix and count_tokens(prefix) + count_tokens(chunk) <= self.size + self.overlap:
                out.append(f"{prefix} {chunk}".strip())
            else:
                out.append(chunk)
        return out

    def _overlap_tail(self, chunk: str) -> str:
        if not chunk:
            return ""
        pieces = [p.strip() for p in _SENTENCE_RE.split(chunk) if p.strip()]
        tail: list[str] = []
        tokens = 0
        for piece in reversed(pieces):
            if tokens >= self.overlap:
                break
            tail.append(piece)
            tokens += count_tokens(piece)
        return " ".join(reversed(tail))

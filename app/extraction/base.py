from dataclasses import dataclass
from typing import Protocol


@dataclass
class PageSegment:
    number: int | None
    text: str


class Extractor(Protocol):
    async def extract(self, data: bytes) -> list[PageSegment]: ...


def get_extractor(content_type: str) -> Extractor:
    from app.extraction.pdf_extractor import PdfExtractor
    from app.extraction.txt_extractor import TxtExtractor

    if content_type == "application/pdf":
        return PdfExtractor()
    if content_type in ("text/plain", "application/octet-stream"):
        return TxtExtractor()
    raise ValueError(f"Unsupported content type: {content_type}")

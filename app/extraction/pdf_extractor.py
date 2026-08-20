from io import BytesIO

from pypdf import PdfReader

from app.extraction.base import PageSegment


class PdfExtractor:
    async def extract(self, data: bytes) -> list[PageSegment]:
        reader = PdfReader(BytesIO(data))
        segments: list[PageSegment] = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                segments.append(PageSegment(number=number, text=text))
        return segments

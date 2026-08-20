from app.extraction.base import PageSegment


class TxtExtractor:
    async def extract(self, data: bytes) -> list[PageSegment]:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return []
        return [PageSegment(number=None, text=text)]

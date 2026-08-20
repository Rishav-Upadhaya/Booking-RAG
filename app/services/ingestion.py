from sqlalchemy import delete

from app.chunking.base import get_chunker
from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.embeddings.base import resolve_embedder
from app.extraction.base import get_extractor
from app.vectorstore.pinecone_store import PineconeVectorStore, sparse_vectors


def _vector_id(doc_id: int, chunk_index: int) -> str:
    return f"doc_{doc_id}_{chunk_index}"


async def process_upload(document_id: int, file_bytes: bytes) -> None:
    async with SessionLocal() as session:
        document = await session.get(Document, document_id)
        document.status = "processing"
        await session.commit()
        try:
            extractor = get_extractor(document.content_type)
            chunker = get_chunker(document.chunking_strategy)
            embedder = resolve_embedder()
            store = PineconeVectorStore()

            segments = await extractor.extract(file_bytes)
            if not segments:
                raise ValueError("no text extracted from file")
            pieces = await chunker.chunk(segments)
            if not pieces:
                raise ValueError("no chunks produced from extracted text")
            chunks = [piece.text for piece in pieces]

            dense_vectors = await embedder.embed(chunks)
            sparse_vector_list = await sparse_vectors(chunks, input_type="passage")

            for index, (piece, dense, sparse) in enumerate(
                zip(pieces, dense_vectors, sparse_vector_list)
            ):
                vid = _vector_id(document_id, index)
                metadata = {
                    "document_id": str(document_id),
                    "chunk_index": str(index),
                    "text": piece.text,
                    "filename": document.filename,
                }
                if piece.page is not None:
                    metadata["page"] = str(piece.page)
                await store.upsert(
                    vector_id=vid,
                    dense=dense,
                    sparse=sparse,
                    metadata=metadata,
                )
                session.add(
                    Chunk(
                        document_id=document_id,
                        chunk_index=index,
                        text=piece.text,
                        vector_id=vid,
                        filename=document.filename,
                        page=piece.page,
                    )
                )

            document.chunk_count = len(chunks)
            document.status = "ready"
        except Exception as exc:  # noqa: BLE001
            document.status = "failed"
            document.error = str(exc)
        await session.commit()


async def delete_document(document_id: int) -> None:
    async with SessionLocal() as session:
        await PineconeVectorStore().delete_by_document(document_id)
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        document = await session.get(Document, document_id)
        if document:
            await session.delete(document)
        await session.commit()

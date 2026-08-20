from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.chunking.base import ChunkingStrategy
from app.core.config import settings
from app.db.models import Chunk, Document
from app.db.session import get_session
from app.schemas.document import ChunkOut, DocumentOut, DocumentUploadOut
from app.services.ingestion import delete_document, process_upload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=202, response_model=DocumentUploadOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(default=ChunkingStrategy.FIXED),
) -> DocumentUploadOut:
    if file.content_type not in (
        "application/pdf",
        "text/plain",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400, detail=f"unsupported content type: {file.content_type}"
        )

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file too large: {len(data)} bytes (limit {settings.max_upload_bytes})",
        )
    document = Document(
        filename=file.filename or "unnamed",
        content_type=file.content_type,
        status="pending",
        chunking_strategy=chunking_strategy,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    background_tasks.add_task(process_upload, document.id, data)
    return DocumentUploadOut(id=document.id)


@router.get("/{document_id}/status", response_model=DocumentOut)
async def get_status(
    document_id: int, session: AsyncSession = Depends(get_session)
) -> Document:
    document = await session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
async def get_chunks(
    document_id: int, session: AsyncSession = Depends(get_session)
) -> list[Chunk]:
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="document not found")
    return document.chunks


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.id.desc()))
    return list(result.scalars().all())


@router.delete("/{document_id}", status_code=204)
async def delete_document_endpoint(document_id: int) -> None:
    await delete_document(document_id)

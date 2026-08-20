import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health
from app.embeddings.base import resolve_embedder
from app.rag.graph import build_agent
from app.vectorstore.pinecone_store import ensure_index

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedder = await asyncio.to_thread(resolve_embedder)
    probe = await embedder.embed(["palm-rag startup self-test"])
    if len(probe) != 1 or len(probe[0]) != embedder.dimension:
        raise RuntimeError("embedder self-test failed: unexpected embedding shape")
    ensure_index(embedder.dimension)
    build_agent()
    logger.info(
        "[startup] embedder=%s dim=%d, self-test passed",
        type(embedder).__name__,
        embedder.dimension,
    )
    yield


app = FastAPI(title="palm-rag", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)

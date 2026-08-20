from fastapi import APIRouter
from sqlalchemy import text

from app.db.session import SessionLocal
from app.memory.redis_memory import get_client

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    async with SessionLocal() as session:
        await session.execute(text("SELECT 1"))
    await get_client().ping()
    return {"status": "ok"}

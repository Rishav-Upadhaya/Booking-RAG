import json
from datetime import UTC, datetime, timedelta

import redis.asyncio as aioredis

from app.core.config import settings

_client: aioredis.Redis | None = None


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


def _key(session_id: str) -> str:
    return f"chat:{session_id}"


class RedisMemory:
    async def replace_history(self, session_id: str, turns: list[dict]) -> None:
        await get_client().set(
            _key(session_id), json.dumps(turns), ex=settings.memory_ttl_seconds
        )

    async def get_history(
        self, session_id: str, limit: int | None = None
    ) -> list[dict]:
        raw = await get_client().get(_key(session_id))
        history = json.loads(raw) if raw else []
        return history[-limit:] if limit else history

    async def clear(self, session_id: str) -> None:
        await get_client().delete(_key(session_id))

    async def truncate(self, session_id: str, keep: int) -> None:
        history = await self.get_history(session_id)
        await self.replace_history(session_id, history[: max(keep, 0)])

    async def list_sessions(self) -> list[dict]:
        client = get_client()
        sessions: list[dict] = []
        async for key in client.scan_iter(match="chat:*", count=200):
            session_id = key.removeprefix("chat:")
            raw = await client.get(key)
            if not raw:
                continue
            try:
                turns = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not turns:
                continue
            ttl = await client.ttl(key)
            updated_at = None
            if ttl is not None and ttl > 0:
                updated_at = datetime.now(UTC) - timedelta(seconds=ttl)
            sessions.append(
                {
                    "session_id": session_id,
                    "turn_count": len(turns),
                    "preview": str(turns[0].get("content", ""))[:120],
                    "updated_at": updated_at,
                }
            )
        sessions.sort(
            key=lambda s: s["updated_at"] or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return sessions

    async def get_json(self, key: str) -> str | None:
        return await get_client().get(key)

    async def set_json(self, key: str, value: str, ttl: int | None = None) -> None:
        await get_client().set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await get_client().delete(key)

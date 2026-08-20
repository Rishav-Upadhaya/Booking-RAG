import json
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk

from app.memory.redis_memory import RedisMemory
from app.rag.graph import default_agent, extract_reply_and_usage


class ChatServiceError(RuntimeError):
    pass


class ChatService:
    def __init__(self, memory: RedisMemory | None = None) -> None:
        self.memory = memory or RedisMemory()

    def _invoke_input(self, session_id: str, message: str) -> dict:
        return {
            "messages": [],
            "session_id": session_id,
            "pending_message": message,
        }

    async def send_message(self, session_id: str, message: str) -> dict:
        agent = default_agent()
        try:
            result = await agent.ainvoke(
                self._invoke_input(session_id, message),
                config={"configurable": {"thread_id": session_id}},
            )
        except Exception as exc:
            raise ChatServiceError(str(exc)) from exc

        reply, usage = extract_reply_and_usage(result["messages"])
        sources = result.get("sources") or []
        return {
            "session_id": session_id,
            "reply": reply,
            "sources": sources,
            "usage": usage,
        }

    async def stream_message(
        self, session_id: str, message: str
    ) -> AsyncIterator[str]:
        agent = default_agent()
        config = {"configurable": {"thread_id": session_id}}

        sources: list = []
        reply_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        seen_usage = False

        try:
            async for mode, payload in agent.astream(
                self._invoke_input(session_id, message),
                config=config,
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom":
                    if isinstance(payload, dict) and "sources" in payload:
                        sources = payload["sources"]
                        yield _sse("sources", {"sources": sources})
                    continue

                if mode != "messages":
                    continue

                chunk, metadata = payload
                if metadata.get("langgraph_node") != "model":
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue

                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    seen_usage = True
                    input_tokens += int(usage.get("input_tokens") or 0)
                    output_tokens += int(usage.get("output_tokens") or 0)

                text = _chunk_text(chunk)
                if text:
                    reply_parts.append(text)
                    yield _sse("delta", {"text": text})

            reply = "".join(reply_parts).strip() or "I could not answer that."
            usage_out = (
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                if seen_usage
                else None
            )
            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "reply": reply,
                    "sources": sources,
                    "usage": usage_out,
                },
            )
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"detail": str(exc)})

    async def history(self, session_id: str, limit: int | None = None) -> list[dict]:
        return await self.memory.get_history(session_id, limit=limit or 10)

    async def rewind(self, session_id: str, keep: int) -> list[dict]:
        await self.memory.truncate(session_id, keep)
        return await self.memory.get_history(session_id)

    async def sessions(self) -> list[dict]:
        return await self.memory.list_sessions()

    async def delete(self, session_id: str) -> None:
        await self.memory.clear(session_id)


def _chunk_text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

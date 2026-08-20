import logging
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph.message import add_messages

from app.booking.extractor import book_interview, process_booking_turn
from app.core.config import settings
from app.core.llm import get_model
from app.intent.classifier import (
    Intent,
    classify,
    has_booking_fragments,
    has_name_intro,
    reply_for,
    seems_booking_conversation,
)
from app.memory.redis_memory import RedisMemory
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    pending_message: str
    sources: list[dict[str, Any]]
    intent: NotRequired[Intent]


SYSTEM_PROMPT = """You are a helpful assistant for a document knowledge base.
- For questions about the documents, answer only from the retrieved context provided. If the context does not contain the answer, say you do not know rather than guessing.
- You also have the conversation history. Use it to answer questions about the conversation itself, such as recalling a previous question, repeating something the user said earlier, or remembering a fact the user shared (their name, preferences, etc.).
- If the user wants to book an interview, use the book_interview tool."""


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _usage_to_dict(usage: Any) -> dict[str, int] | None:
    if not usage:
        return None
    try:
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    except (TypeError, ValueError):
        return None


def extract_reply_and_usage(messages: list[BaseMessage]) -> tuple[str, dict[str, int] | None]:
    reply = ""
    usage: dict[str, int] | None = None
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            reply = _text_of(message).strip()
        usage = _usage_to_dict(getattr(message, "usage_metadata", None))
        if reply:
            break
    if not reply:
        reply = "I could not answer that."
    return reply, usage


class RetrievalMiddleware(AgentMiddleware):
    def __init__(self, retriever: Retriever, top_k: int | None = None) -> None:
        self.retriever = retriever
        self.top_k = top_k or settings.top_k

    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any]:
        writer = get_stream_writer()
        if state.get("intent") != Intent.DOMAIN:
            writer({"sources": []})
            return {"sources": []}

        try:
            chunks = await self.retriever.retrieve(state["pending_message"], top_k=self.top_k)
            context = "\n\n".join(
                f"[chunk {i + 1}] {chunk.text}"
                + f" (document {chunk.document_id}"
                + (f", file {chunk.filename}" if chunk.filename else "")
                + (f", page {chunk.page}" if chunk.page is not None else "")
                + ")"
                for i, chunk in enumerate(chunks)
            )
            sources = [
                {
                    "text": chunk.text,
                    "score": chunk.score,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                }
                for chunk in chunks
            ]
            writer({"sources": sources})
            return {
                "sources": sources,
                "messages": [*state["messages"], SystemMessage(content=f"Retrieved context:\n{context}")],
            }
        except Exception:  # noqa: BLE001
            writer({"sources": []})
            return {"sources": []}


class MemoryMiddleware(AgentMiddleware):
    def __init__(self, memory: RedisMemory) -> None:
        self.memory = memory

    def _resolve_session_id(self, state: AgentState) -> str:
        return state.get("session_id") or "default"

    async def abefore_agent(self, state: AgentState, runtime: Any) -> dict[str, Any]:
        session_id = self._resolve_session_id(state)
        history = await self.memory.get_history(session_id, limit=settings.max_memory_turns)
        messages = [
            HumanMessage(content=turn["content"])
            if turn["role"] == "user"
            else AIMessage(content=turn["content"])
            for turn in history
            if turn["role"] in ("user", "assistant")
            and turn.get("type") in (None, "human", "ai")
            and str(turn.get("content", "")).strip()
        ]
        messages.append(HumanMessage(content=state["pending_message"]))
        return {"messages": messages}

    async def aafter_agent(self, state: AgentState, runtime: Any) -> dict[str, Any]:
        session_id = self._resolve_session_id(state)
        turns: list[dict[str, str]] = []
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            content = _text_of(message).strip()
            if not content:
                continue
            turns.append({"role": role, "content": content})
        await self.memory.replace_history(session_id, turns)
        return {}


class IntentMiddleware(AgentMiddleware):
    def __init__(self, memory: RedisMemory) -> None:
        self.memory = memory

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any]:
        session_id = state.get("session_id") or "default"
        partial_raw = await self.memory.get_json(f"booking:partial:{session_id}")
        intent = await classify(state["pending_message"], has_partial_booking=bool(partial_raw))
        message = state["pending_message"]
        if intent != Intent.BOOKING:
            prior_user_text = [
                _text_of(m)
                for m in state.get("messages", [])
                if isinstance(m, HumanMessage)
            ]
            in_booking_context = partial_raw or any(
                seems_booking_conversation(t) for t in prior_user_text
            )
            if has_booking_fragments(message) or (
                in_booking_context and has_name_intro(message)
            ):
                intent = Intent.BOOKING
        if intent in (Intent.GREETING, Intent.GUARDRAIL, Intent.OFF_TOPIC):
            return {
                "intent": intent,
                "messages": [*state["messages"], AIMessage(content=reply_for(intent))],
                "jump_to": "end",
            }
        return {"intent": intent}


class BookingMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any]:
        if state.get("intent") != Intent.BOOKING:
            return {}
        session_id = state.get("session_id") or "default"
        try:
            reply = await process_booking_turn(session_id, state["pending_message"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("booking turn failed: %s", exc)
            reply = "Sorry, I could not process the booking. Please try again."
        return {
            "messages": [*state["messages"], AIMessage(content=reply)],
            "jump_to": "end",
        }


_agent: Any | None = None


def build_agent() -> Any:
    global _agent
    middleware = [
        MemoryMiddleware(memory=RedisMemory()),
        IntentMiddleware(memory=RedisMemory()),
        BookingMiddleware(),
        RetrievalMiddleware(retriever=Retriever()),
        PIIMiddleware("credit_card", strategy="redact"),
        PIIMiddleware("url", strategy="redact"),
        PIIMiddleware("ip", strategy="redact"),
        SummarizationMiddleware(model=get_model(), trigger=("fraction", 0.6), keep=("fraction", 0.3)),
    ]
    _agent = create_agent(
        model=get_model(),
        tools=[book_interview],
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        state_schema=AgentState,
    )
    return _agent


def default_agent() -> Any:
    if _agent is None:
        raise RuntimeError("agent not built yet; call build_agent() during app startup")
    return _agent
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.booking.validator import BookingFields
from app.db.models import Booking
from app.db.session import get_session
from app.schemas.chat import (
    BookingIn,
    BookingOut,
    MessageIn,
    MessageOut,
    RewindIn,
    RewindOut,
    SessionOut,
    TurnOut,
)
from app.services.bookings import persist_booking
from app.services.chat_service import ChatService, ChatServiceError

router = APIRouter(tags=["chat"])


@router.post("/chat/message", response_model=MessageOut)
async def chat_message(payload: MessageIn) -> dict:
    try:
        return await ChatService().send_message(payload.session_id, payload.message)
    except ChatServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"the assistant is temporarily unavailable: {exc}",
        ) from exc


@router.post("/chat/stream")
async def chat_stream(payload: MessageIn) -> StreamingResponse:
    service = ChatService()
    return StreamingResponse(
        service.stream_message(payload.session_id, payload.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/sessions", response_model=list[SessionOut])
async def list_sessions() -> list[dict]:
    return await ChatService().sessions()


@router.get("/chat/{session_id}/history", response_model=list[TurnOut])
async def chat_history(session_id: str, limit: int | None = None) -> list[dict]:
    return await ChatService().history(session_id, limit=limit)


@router.post("/chat/{session_id}/rewind", response_model=RewindOut)
async def chat_rewind(session_id: str, payload: RewindIn) -> dict:
    history = await ChatService().rewind(session_id, payload.keep)
    return {"session_id": session_id, "history": history}


@router.delete("/chat/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    await ChatService().delete(session_id)


@router.post("/bookings", response_model=BookingOut)
async def create_booking(payload: BookingIn) -> Booking:
    try:
        fields = BookingFields.model_validate(payload.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await persist_booking(payload.session_id, fields)


@router.get("/bookings/{booking_id}", response_model=BookingOut)
async def get_booking(
    booking_id: int, session: AsyncSession = Depends(get_session)
) -> Booking:
    booking = await session.get(Booking, booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="booking not found")
    return booking

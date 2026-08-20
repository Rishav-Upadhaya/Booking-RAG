from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.document import ORMModel


class MessageIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)
    message: str = Field(min_length=1, max_length=8000)


class RewindIn(BaseModel):
    keep: int = Field(ge=0, description="Number of turns to keep from the start")


class SourceOut(BaseModel):
    text: str
    score: float | None = None
    document_id: int
    filename: str | None = None
    page: int | None = None


class UsageOut(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class MessageOut(BaseModel):
    session_id: str
    reply: str
    sources: list[SourceOut]
    usage: UsageOut | None = None


class TurnOut(BaseModel):
    role: str
    content: str


class RewindOut(BaseModel):
    session_id: str
    history: list[TurnOut]


class SessionOut(BaseModel):
    session_id: str
    turn_count: int
    preview: str
    updated_at: datetime | None = None


class BookingIn(BaseModel):
    session_id: str
    name: str
    email: str
    date: str
    time: str


class BookingOut(ORMModel):
    id: int
    session_id: str
    name: str
    email: str
    date: str
    time: str

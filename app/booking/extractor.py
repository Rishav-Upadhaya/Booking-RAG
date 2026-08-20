import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain.tools import ToolRuntime, tool
from pydantic import ValidationError

from app.booking.validator import (
    BookingFields,
    _normalize_date_text,
)
from app.core.config import settings
from app.core.llm import get_model
from app.memory.redis_memory import RedisMemory
from app.services.bookings import persist_booking

MISSING_FIELD_QUESTIONS = {
    "name": "What name should I put on the booking?",
    "email": "What email address should I use for the booking?",
    "date": "Which date works for you?",
    "time": "What time would you prefer?",
}

PARTIAL_KEY = "booking:partial:{session_id}"
ASKED_KEY = "booking:asked:{session_id}"
_FIELD_ORDER = ("name", "email", "date", "time")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_NAME_RE = re.compile(
    r"(?:my name is|i am called|the name is|name is|call me|name\s*[:=])"
    r"\s+([A-Za-z][A-Za-z .'’-]{1,60})",
    re.IGNORECASE,
)
_MONTHS = (
    r"jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december"
)
_DATE_RE = re.compile(
    r"\d{1,4}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:\d{1,2})(?:st|nd|rd|th)?[\s-](?:" + _MONTHS + r")(?:[\s,]+(?:\d{2,4}))?"
    r"|(?:" + _MONTHS + r")\s+(?:\d{1,2})(?:st|nd|rd|th)?(?:[\s,]+(?:\d{2,4}))?",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s?[ap]m\b|\b\d{1,2}:\d{2}\b", re.IGNORECASE
)

Normalizer = Callable[[str, str], Awaitable[str]]
_normalizer: Normalizer | None = None


def set_normalizer(fn: Normalizer) -> None:
    global _normalizer
    _normalizer = fn


async def _llm_normalize(which: str, raw: str) -> str:
    target = "date as YYYY-MM-DD" if which == "date" else "time as 24h HH:MM"
    out = await get_model().ainvoke(
        [
            (
                "system",
                (
                    f"You are a data normalizer. Reply with ONLY the {target} for "
                    f"the value the user gives. If you cannot, reply with exactly UNKNOWN."
                ),
            ),
            ("user", raw),
        ]
    )
    value = (str(out.content if hasattr(out, "content") else out)).strip()
    return _canonical(value, which) or ""


async def _normalize(field: str, raw: str) -> str:
    if _normalizer is not None:
        return await _normalizer(field, raw)  # type: ignore[misc]
    return await _llm_normalize(field, raw)


def _canonical(value: str, which: str) -> str | None:
    try:
        if which == "date":
            value = _normalize_date_text(value)
            kwargs: dict[str, str] = {which: value}
            return BookingFields(**kwargs).date
        return BookingFields(time=value).time
    except ValidationError:
        return None


@dataclass
class RawBooking:
    name: str = ""
    email: str = ""
    date: str = ""
    time: str = ""
    errors: list[str] = field(default_factory=list)


def extract_booking_fields(text: str) -> RawBooking:
    name_match = _NAME_RE.search(text)
    email_match = _EMAIL_RE.search(text)
    date_match = _DATE_RE.search(text)
    time_match = _TIME_RE.search(text)
    name = name_match.group(1).strip().strip(".,;!?") if name_match else ""
    if not name:
        name = _trailing_name(text, email_match, date_match, time_match)
    raw = RawBooking(
        name=name,
        email=email_match.group(0) if email_match else "",
        date=date_match.group(0) if date_match else "",
        time=time_match.group(0) if time_match else "",
    )
    for value, which in (
        (raw.date, "date"),
        (raw.time, "time"),
    ):
        if value:
            raw.errors.extend(_field_errors(value, which))
    return raw


_STOPWORDS = {
    "a", "an", "the", "on", "at", "for", "by", "please", "with", "my", "is",
    "and", "or", "of", "to", "in", "it", "from", "i", "am", "booking",
    "interview", "appointment", "schedule", "book",
}


def _trailing_name(
    text: str, email_match, date_match, time_match
) -> str:
    matches = [m for m in (email_match, date_match, time_match) if m]
    if not matches:
        return ""
    tail = text[matches[-1].end() :]
    words = re.split(r"\s*[,;]\s*|\s+", tail.strip())
    names = [
        w
        for w in words
        if w
        and w.lower() not in _STOPWORDS
        and re.fullmatch(r"[A-Za-z][A-Za-z.'’\-]*", w)
    ]
    return " ".join(names) if 1 <= len(names) <= 3 else ""


def _field_errors(value: str, which: str) -> list[str]:
    if _canonical(value, which):
        return []
    return [
        (
            f'I could not understand the {which} "{value}". '
            "Use YYYY-MM-DD for the date and 24h HH:MM (or '2pm') for the time."
        )
    ]


async def process_booking_turn(session_id: str, text: str) -> str:
    raw = extract_booking_fields(text)
    extracted_any = bool(raw.name or raw.email or raw.date or raw.time)
    fields = BookingFields(name=raw.name, email=raw.email, date="", time="")
    errors = list(raw.errors)

    for value, which in ((raw.date, "date"), (raw.time, "time")):
        if not value:
            continue
        canonical = _canonical(value, which)
        if canonical is None:
            canonical = await _normalize(which, value)
        if canonical:
            fields = fields.model_copy(update={which: canonical})
        else:
            errors.append(_field_errors(value, which)[0])

    asked = await _asked_field(session_id)
    if not extracted_any and asked and not getattr(fields, asked) and len(text.split()) <= 6:
        value = text.strip().strip(".,;!?")
        canonical = _validate_positional(asked, value)
        if canonical is None:
            canonical = await _normalize(asked, value)
        if canonical:
            fields = fields.model_copy(update={asked: canonical})
        elif asked in ("date", "time"):
            errors.append(_field_errors(value, asked)[0])

    if errors:
        valid = BookingFields(
            name=raw.name, email=raw.email, date=fields.date, time=fields.time
        )
        await merge_and_persist(session_id, valid)
        return errors[0]

    msg, completed = await merge_and_persist(session_id, fields)
    await _update_asked(session_id, completed=completed)
    return msg


def _validate_positional(which: str, value: str) -> str | None:
    if which == "name":
        return value if value and len(value) <= 60 else None
    return _canonical(value, which)


async def _asked_field(session_id: str) -> str:
    raw = await RedisMemory().get_json(ASKED_KEY.format(session_id=session_id))
    return raw if raw in _FIELD_ORDER else ""


async def _update_asked(session_id: str, *, completed: bool) -> None:
    memory = RedisMemory()
    key = ASKED_KEY.format(session_id=session_id)
    if completed:
        await memory.delete(key)
        return
    partial_raw = await memory.get_json(PARTIAL_KEY.format(session_id=session_id))
    existing = BookingFields.model_validate(json.loads(partial_raw)) if partial_raw else BookingFields()
    next_field = next((f for f in _FIELD_ORDER if not getattr(existing, f)), "")
    if next_field:
        await memory.set_json(key, next_field, ttl=settings.memory_ttl_seconds)
    else:
        await memory.delete(key)


async def merge_and_persist(
    session_id: str, fields: BookingFields, persist=None
) -> tuple[str, bool]:
    if persist is None:
        persist = persist_booking
    memory = RedisMemory()
    partial = PARTIAL_KEY.format(session_id=session_id)
    raw = await memory.get_json(partial)
    existing = BookingFields.model_validate(json.loads(raw)) if raw else BookingFields()

    merged = BookingFields(
        name=fields.name or existing.name,
        email=fields.email or existing.email,
        date=fields.date or existing.date,
        time=fields.time or existing.time,
    )
    if not merged.complete:
        await memory.set_json(
            partial, json.dumps(merged.model_dump()), ttl=settings.memory_ttl_seconds
        )
        return ask_for_missing(merged), False

    booking = await persist(session_id, merged)
    await memory.delete(partial)
    return (
        (
            f"Booking #{booking.id} confirmed: {merged.name} ({merged.email}) "
            f"on {merged.date} at {merged.time}."
        ),
        True,
    )


def ask_for_missing(fields: BookingFields) -> str:
    for key in _FIELD_ORDER:
        if not getattr(fields, key):
            return MISSING_FIELD_QUESTIONS[key]
    return "All set, what details would you like?"


@tool(description="Book an interview for the user. Call this whenever the user asks to book an interview or provides any booking details (name, email, date, time). Fill in only the values you know and leave the rest as empty strings.")
async def book_interview(
    name: str = "",
    email: str = "",
    date: str = "",
    time: str = "",
    *,
    runtime: ToolRuntime,
) -> str:
    session_id = (runtime.config.get("configurable") or {}).get("thread_id", "default")
    try:
        fields = BookingFields(name=name, email=email, date=date, time=time)
    except ValidationError as exc:
        return (
            f"Some of those details did not look right: {exc}. Could you recheck them?"
        )
    reply, _ = await merge_and_persist(session_id, fields)
    return reply
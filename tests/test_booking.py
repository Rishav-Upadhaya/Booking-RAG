"""Booking flow: field parsing, validation and the multi-turn slot filling.

Redis and Postgres are replaced with in-memory fakes, and the LLM normalizer
with a stub, so these run without any services.
"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.booking import extractor
from app.booking.validator import BookingFields


# ── BookingFields ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-10-05", "2026-10-05"),
        ("05/10/2026", "2026-10-05"),        # day-first wins over month-first
        ("5th October 2026", "2026-10-05"),
        ("Sept 5 2026", "2026-09-05"),
        ("October 5", f"{datetime.now(UTC).year}-10-05"),  # year defaults to now
    ],
)
def test_dates_normalize_to_iso(raw, expected):
    assert BookingFields(date=raw).date == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("14:30", "14:30"), ("2:30 pm", "14:30"), ("2pm", "14:00"), ("9 AM", "09:00")],
)
def test_times_normalize_to_24h(raw, expected):
    assert BookingFields(time=raw).time == expected


@pytest.mark.parametrize("field,value", [("date", "next someday"), ("time", "noonish"), ("email", "not-an-email")])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        BookingFields(**{field: value})


def test_complete_requires_all_four_fields():
    assert not BookingFields(name="A", email="a@b.co", date="2026-10-05").complete
    assert BookingFields(name="A", email="a@b.co", date="2026-10-05", time="10:00").complete


# ── extract_booking_fields ────────────────────────────────────────────────────

def test_extracts_every_field_from_one_sentence():
    raw = extractor.extract_booking_fields(
        "My name is Sita Sharma, email sita@example.com, on 5th October 2026 at 2pm"
    )
    assert (raw.name, raw.email, raw.date, raw.time) == (
        "Sita Sharma", "sita@example.com", "5th October 2026", "2pm"
    )
    assert raw.errors == []


def test_full_month_name_keeps_the_year():
    # Regression: "October" used to match as "Oct", dropping "2027".
    raw = extractor.extract_booking_fields("on 5th October 2027 at 2pm")
    assert raw.date == "5th October 2027"
    assert BookingFields(date=raw.date).date == "2027-10-05"


def test_name_after_the_other_fields_is_picked_up():
    raw = extractor.extract_booking_fields("sita@example.com 2026-10-05 10:00 Sita Sharma")
    assert raw.name == "Sita Sharma"


def test_unparseable_date_is_reported_not_guessed():
    raw = extractor.extract_booking_fields("book me on 2026-13-45 at 10:00")
    assert raw.date == "2026-13-45"
    assert raw.errors and "date" in raw.errors[0]


# ── process_booking_turn (multi-turn) ─────────────────────────────────────────

class FakeRedis:
    store: dict = {}

    async def get_json(self, key):
        return self.store.get(key)

    async def set_json(self, key, value, ttl=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture()
def booking_env(monkeypatch):
    FakeRedis.store = {}
    saved = []

    async def fake_persist(session_id, fields):
        saved.append((session_id, fields))
        return SimpleNamespace(id=len(saved))

    async def no_llm(field, raw):
        return ""  # the LLM fallback "could not normalize"

    monkeypatch.setattr(extractor, "RedisMemory", FakeRedis)
    monkeypatch.setattr(extractor, "persist_booking", fake_persist)
    monkeypatch.setattr(extractor, "_normalizer", no_llm)
    return saved


def turn(text, session="s1"):
    return asyncio.run(extractor.process_booking_turn(session, text))


def test_one_shot_booking_is_confirmed_and_persisted(booking_env):
    reply = turn("My name is Sita Sharma, sita@example.com, 2026-10-05 at 2pm")
    assert reply.startswith("Booking #1 confirmed")
    [(session, fields)] = booking_env
    assert (session, fields.date, fields.time) == ("s1", "2026-10-05", "14:00")
    assert FakeRedis.store == {}  # partial state and pending question cleared


def test_slot_filling_asks_for_each_missing_field_in_order(booking_env):
    assert turn("I'd like to book an interview") == extractor.MISSING_FIELD_QUESTIONS["name"]
    assert turn("Sita Sharma") == extractor.MISSING_FIELD_QUESTIONS["email"]
    assert turn("sita@example.com") == extractor.MISSING_FIELD_QUESTIONS["date"]
    assert turn("2026-10-05") == extractor.MISSING_FIELD_QUESTIONS["time"]
    assert turn("10am").startswith("Booking #1 confirmed: Sita Sharma (sita@example.com) on 2026-10-05 at 10:00")
    assert len(booking_env) == 1


def test_bad_date_keeps_valid_fields_and_explains(booking_env):
    reply = turn("My name is Sita Sharma, sita@example.com, 2026-13-45")
    assert "could not understand the date" in reply
    assert booking_env == []
    # Name and email survived; the next turn only needs date and time.
    assert turn("2026-10-05 at 10:00").startswith("Booking #1 confirmed: Sita Sharma")


def test_llm_normalizer_is_used_when_rules_cannot_parse(booking_env, monkeypatch):
    async def llm(field, raw):
        return {"date": "2026-10-06", "time": "09:30"}[field]

    monkeypatch.setattr(extractor, "_normalizer", llm)
    turn("My name is Sita Sharma, sita@example.com")
    turn("tomorrow")  # the question was "which date"; rules fail, LLM resolves it
    assert turn("half nine in the morning").endswith("on 2026-10-06 at 09:30.")


def test_sessions_do_not_share_partial_bookings(booking_env):
    turn("My name is Sita Sharma", session="a")
    assert turn("sita@example.com", session="b") == extractor.MISSING_FIELD_QUESTIONS["name"]

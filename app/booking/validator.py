import re
from datetime import UTC, datetime

from pydantic import BaseModel, EmailStr, TypeAdapter, field_validator

_EMAIL_ADAPTER = TypeAdapter(EmailStr)

_ORDINAL_RE = re.compile(r"(\d)(?:st|nd|rd|th)\b", re.IGNORECASE)

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%B %d %Y",
    "%d %b",
    "%b %d",
    "%d %B",
    "%B %d",
    "%d %b %y",
    "%b %d %y",
)
_TIME_FORMATS = ("%H:%M", "%I:%M %p", "%I:%M%p", "%I%p", "%I %p")


def _normalize_date_text(value: str) -> str:
    value = _ORDINAL_RE.sub(r"\1", value)
    value = re.sub(r"\bsept\b", "sep", value, flags=re.IGNORECASE)
    return value


class BookingFields(BaseModel):
    name: str = ""
    email: str = ""
    date: str = ""
    time: str = ""

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if value:
            _EMAIL_ADAPTER.validate_python(value)
        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        if not value:
            return value
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(_normalize_date_text(value), fmt)  # noqa: DTZ007
            except ValueError:
                continue
            if parsed.year == 1900:
                parsed = parsed.replace(year=datetime.now(UTC).year)
            return parsed.strftime("%Y-%m-%d")
        raise ValueError(f"unparseable date: {value}")

    @field_validator("time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not value:
            return value
        for fmt in _TIME_FORMATS:
            try:
                return datetime.strptime(value, fmt).strftime("%H:%M")  # noqa: DTZ007
            except ValueError:
                continue
        raise ValueError(f"unparseable time: {value}")

    @property
    def complete(self) -> bool:
        return all((self.name, self.email, self.date, self.time))

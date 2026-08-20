from app.booking.validator import BookingFields
from app.db.models import Booking
from app.db.session import SessionLocal


async def persist_booking(session_id: str, fields: BookingFields) -> Booking:
    async with SessionLocal() as session:
        booking = Booking(
            session_id=session_id,
            name=fields.name,
            email=fields.email,
            date=fields.date,
            time=fields.time,
        )
        session.add(booking)
        await session.commit()
        await session.refresh(booking)
        return booking

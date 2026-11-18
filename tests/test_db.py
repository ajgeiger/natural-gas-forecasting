from datetime import date

from sqlalchemy import func, select

from natgas.db import Price, session_scope, upsert


def price_row(day: date, high: float) -> dict:
    return {
        "date": day,
        "open": 3.0,
        "high": high,
        "low": 2.9,
        "close": 3.05,
        "volume": 1000,
        "source": "yahoo",
    }


def test_upsert_is_idempotent_and_updates(settings):
    rows = [price_row(date(2024, 1, 2), 3.1), price_row(date(2024, 1, 3), 3.2)]
    with session_scope(settings.db_url) as session:
        upsert(session, Price, rows)
    with session_scope(settings.db_url) as session:
        upsert(session, Price, [price_row(date(2024, 1, 3), 3.5)])

    with session_scope(settings.db_url) as session:
        assert session.scalar(select(func.count()).select_from(Price)) == 2
        assert session.get(Price, date(2024, 1, 3)).high == 3.5
        assert session.get(Price, date(2024, 1, 3)).ingested_at is not None


def test_upsert_empty_rows(settings):
    with session_scope(settings.db_url) as session:
        assert upsert(session, Price, []) == 0

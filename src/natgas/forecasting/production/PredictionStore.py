"""Append-only table of live forecasts."""

from __future__ import annotations

from sqlalchemy import func, insert, select

from natgas.db import Prediction, session_scope


class PredictionStore:
    """Saves each scoring run to the predictions table and never changes old rows.

    A run is one as-of date and one stage. The table key is as_of_date, stage,
    and horizon_day, so the same run cannot be saved twice.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def has_run(self, as_of_date, stage: str) -> bool:
        query = select(func.count()).where(
            Prediction.as_of_date == as_of_date, Prediction.stage == stage
        )
        with session_scope(self.db_url) as session:
            return session.execute(query).scalar_one() > 0

    def append(self, rows: list[dict]) -> int:
        if rows:
            with session_scope(self.db_url) as session:
                session.execute(insert(Prediction), rows)
        return len(rows)

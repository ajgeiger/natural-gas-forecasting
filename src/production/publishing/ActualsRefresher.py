"""Joins live forecasts to the actual price once their target date has passed."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from backtesting.BacktestReport import HORIZON_BUCKETS
from data.database import ForecastActual, Prediction, Price, session_scope, upsert
from data.database.models import utc_now
from data.provider.ForecastDataProvider import PRICE_COLUMNS

PREDICTION_COLUMNS = [
    "as_of_date",
    "stage",
    "horizon_day",
    "run_id",
    "target_date",
    "model",
    "model_version",
    "forecast",
    "lower",
    "upper",
]


class ActualsRefresher:
    """Fills forecast_actuals with every forecast whose actual price is known.

    A forecast matures when its target date is on or before the last complete
    trading day and the prices table has that day. Every run rewrites the
    matured rows, so a corrected price also corrects the error.
    """

    def __init__(self, db_url: str, target_column: str = "high") -> None:
        if target_column not in PRICE_COLUMNS:
            raise ValueError(f"target_column must be one of {PRICE_COLUMNS}")
        self.db_url = db_url
        self.target_column = target_column

    @staticmethod
    def bucket_for(horizon_day: int) -> str:
        for first, last in HORIZON_BUCKETS:
            if first <= horizon_day <= last:
                return f"days {first} to {last}"
        return f"days {HORIZON_BUCKETS[-1][1] + 1} plus"

    def refresh(self, last_complete_day: date) -> int:
        """Rows written to forecast_actuals."""
        columns = [getattr(Prediction, name) for name in PREDICTION_COLUMNS]
        actual = getattr(Price, self.target_column).label("actual")
        query = (
            select(*columns, actual)
            .join(Price, Price.date == Prediction.target_date)
            .where(Prediction.target_date <= last_complete_day)
        )
        with session_scope(self.db_url) as session:
            matured = session.execute(query).mappings().all()
            now = utc_now()
            rows = [
                {
                    **row,
                    "horizon_bucket": self.bucket_for(row["horizon_day"]),
                    "error": row["forecast"] - row["actual"],
                    "filled_at": now,
                }
                for row in matured
            ]
            return upsert(session, ForecastActual, rows)

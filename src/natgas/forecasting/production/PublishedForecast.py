"""Reads the latest published production forecast and recent prices."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from natgas.db import get_engine
from natgas.forecasting.data.SqlReader import SqlReader

FORECAST_COLUMNS = [
    "as_of_date",
    "target_date",
    "horizon_day",
    "model",
    "model_version",
    "forecast",
    "lower",
    "upper",
]

PRICE_COLUMNS = ["date", "high"]


class PublishedForecast:
    """The newest stage=production forecast, plus recent actual prices for a chart."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def latest(self) -> pd.DataFrame:
        """The 30 day published forecast for the newest as-of date."""
        columns = ", ".join(FORECAST_COLUMNS)
        query = text(
            f"SELECT {columns} FROM predictions "
            "WHERE stage = 'production' "
            "AND as_of_date = ("
            "SELECT MAX(as_of_date) FROM predictions WHERE stage = 'production'"
            ") ORDER BY horizon_day"
        )
        return SqlReader.read(get_engine(self.db_url), query, FORECAST_COLUMNS)

    def prices(self, days: int = 60) -> pd.DataFrame:
        """The most recent daily highs, to plot next to the forecast."""
        query = text("SELECT date, high FROM prices ORDER BY date DESC LIMIT :days")
        frame = SqlReader.read(get_engine(self.db_url), query, PRICE_COLUMNS, {"days": days})
        return frame.sort_values("date").reset_index(drop=True)

"""Cuts the backtest forecasts off where simulated production begins."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, select

from data.database import BacktestForecast, session_scope


class BacktestTrimmer:
    """Removes every backtest forecast made on or after the cutoff.

    The dashboard shows backtests and production side by side. Once a run of
    past days is replayed as production, a backtest from one of those as-of
    dates would show the same day twice. Trimming keeps every as-of date in
    exactly one of the two: before the cutoff it is a backtest, from the
    cutoff on it is production. Both the backtest_forecasts table and its
    Parquet copy are trimmed.
    """

    def __init__(self, db_url: str, parquet_path: str | Path | None = None) -> None:
        self.db_url = db_url
        self.parquet_path = Path(parquet_path) if parquet_path is not None else None

    def trim_table(self, cutoff: date) -> int:
        """Rows deleted from backtest_forecasts."""
        with session_scope(self.db_url) as session:
            count = select(func.count()).where(BacktestForecast.as_of_date >= cutoff)
            removed = session.execute(count).scalar_one()
            session.execute(delete(BacktestForecast).where(BacktestForecast.as_of_date >= cutoff))
        return int(removed)

    def trim_parquet(self, cutoff: date) -> int:
        """Rows dropped from the Parquet file. Zero when there is no file."""
        if self.parquet_path is None or not self.parquet_path.exists():
            return 0
        frame = pd.read_parquet(self.parquet_path)
        keep = pd.to_datetime(frame["as_of_date"]) < pd.Timestamp(cutoff)
        frame[keep].reset_index(drop=True).to_parquet(self.parquet_path, index=False)
        return int((~keep).sum())

    def trim(self, cutoff: date) -> dict[str, int]:
        return {"table": self.trim_table(cutoff), "parquet": self.trim_parquet(cutoff)}

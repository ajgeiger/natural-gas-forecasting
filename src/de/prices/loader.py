"""Daily natural gas futures prices from Yahoo Finance."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.database import Price
from de.utils import Loader


class PriceLoader(Loader):
    name = "prices"
    table = Price
    source = "yahoo"

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV for [start, end]. yfinance treats end as exclusive."""
        return yf.Ticker(self.settings.ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

    def to_rows(self, history: pd.DataFrame) -> list[dict]:
        if history.empty:
            return []
        frame = history.rename(columns=str.lower).dropna(subset=["high"])
        rows = []
        for timestamp, bar in frame.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(timestamp).date(),
                    "open": round(float(bar["open"]), 4),
                    "high": round(float(bar["high"]), 4),
                    "low": round(float(bar["low"]), 4),
                    "close": round(float(bar["close"]), 4),
                    "volume": int(bar["volume"]) if pd.notna(bar["volume"]) else None,
                    "source": self.source,
                }
            )
        return rows

    def rows(self, start: date, end: date) -> list[dict]:
        return self.to_rows(self.fetch(start, end))

"""Daily natural gas futures prices from Yahoo Finance."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from natgas.config import Settings
from natgas.db import Price, session_scope, upsert
from natgas.ingest.runs import tracked_run

log = logging.getLogger(__name__)

SOURCE = "yahoo"


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Return daily OHLCV for [start, end]. yfinance treats end as exclusive."""
    return yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
    )


def to_rows(history: pd.DataFrame) -> list[dict]:
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
                "source": SOURCE,
            }
        )
    return rows


def ingest_prices(settings: Settings, start: date, end: date) -> int:
    with tracked_run(settings.db_url, "prices") as run:
        rows = to_rows(fetch_prices(settings.ticker, start, end))
        if not rows:
            log.warning("no price data returned for %s %s to %s", settings.ticker, start, end)
        with session_scope(settings.db_url) as session:
            run.rows = upsert(session, Price, rows)
    return run.rows

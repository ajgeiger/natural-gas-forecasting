"""Loads the newest prices, weather, and storage before a scoring run."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import func, select

from natgas.config import Settings
from natgas.db import Price, StorageWeekly, WeatherDaily, session_scope
from natgas.de.prices import PriceLoader
from natgas.de.storage import StorageLoader
from natgas.de.weather import WeatherLoader

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 30


class DataRefresher:
    """Runs the existing loaders over the last 30 days, or the full history if a table is empty.

    Reloading 30 days picks up late corrections: Yahoo sometimes revises the
    last bar, and the Open-Meteo archive replaces the recent days that first
    came from the forecast API. The weather loader also stores the forecast
    for the next 14 days, which production scoring uses as future weather.

    Prices and weather must load, or the run stops. Storage needs EIA_API_KEY
    and no model uses it yet, so a storage failure is only logged.
    """

    def __init__(self, settings: Settings, today: date, lookback_days: int = LOOKBACK_DAYS) -> None:
        self.settings = settings
        self.today = today
        self.lookback_days = lookback_days

    def start_for(self, column) -> date:
        with session_scope(self.settings.db_url) as session:
            latest = session.execute(select(func.max(column))).scalar_one()
        if latest is None:
            return self.settings.default_start
        return self.today - timedelta(days=self.lookback_days)

    def refresh(self) -> dict[str, int]:
        """Rows loaded per source."""
        loaded = {
            "prices": PriceLoader(self.settings).run(self.start_for(Price.date), self.today),
            "weather": WeatherLoader(self.settings, self.today).run(
                self.start_for(WeatherDaily.date), self.today
            ),
        }
        try:
            start = self.start_for(StorageWeekly.week_ending)
            loaded["storage"] = StorageLoader(self.settings).run(start, self.today)
        except Exception as error:
            log.warning("storage refresh failed, scoring without it: %s", error)
            loaded["storage"] = 0
        return loaded

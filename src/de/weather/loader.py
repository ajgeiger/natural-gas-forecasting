"""Daily temperatures and degree days from Open-Meteo (no API key needed).

History comes from the archive API, which lags a few days behind. The forecast
API fills the gap from the last archived day through the forecast horizon.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from config import Location, Settings
from data.database import WeatherDaily
from de.utils import Loader, date_chunks, get_json, tracked_run

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The newest archive days are still being written and can fail to load
ARCHIVE_LAG_DAYS = 7
FORECAST_MAX_PAST_DAYS = 90
# Long archive requests sometimes fail mid-stream, so history is pulled a year at a time
CHUNK_DAYS = 366


def degree_days(tavg_f: float, base_temp_f: float) -> tuple[float, float]:
    """Heating and cooling degree days for one day."""
    hdd = max(0.0, base_temp_f - tavg_f)
    cdd = max(0.0, tavg_f - base_temp_f)
    return round(hdd, 2), round(cdd, 2)


def fetch_daily_mean(url: str, location: Location, start: date, end: date) -> dict[date, float]:
    temps: dict[date, float] = {}
    for chunk_start, chunk_end in date_chunks(start, end, CHUNK_DAYS):
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
        }
        daily = get_json(url, params)["daily"]

        # The newest archive days can come back as null until they are processed
        for day, temp in zip(daily["time"], daily["temperature_2m_mean"], strict=True):
            if temp is not None:
                temps[date.fromisoformat(day)] = temp
    return temps


class WeatherLoader(Loader):
    """Observed weather per city, plus the forecast window when the range reaches today."""

    name = "weather"
    table = WeatherDaily

    def __init__(self, settings: Settings, today: date | None = None) -> None:
        super().__init__(settings)
        self.today = today or date.today()

    def location_rows(self, location: Location, start: date, end: date) -> list[dict]:
        temps: dict[date, float] = {}

        archive_end = min(end, self.today - timedelta(days=ARCHIVE_LAG_DAYS))
        if start <= archive_end:
            temps.update(fetch_daily_mean(ARCHIVE_URL, location, start, archive_end))

        # The forecast API only reaches about three months back
        forecast_start = max(temps) + timedelta(days=1) if temps else start
        forecast_start = max(forecast_start, self.today - timedelta(days=FORECAST_MAX_PAST_DAYS))
        if forecast_start <= end:
            temps.update(fetch_daily_mean(FORECAST_URL, location, forecast_start, end))

        rows = []
        for day, tavg_f in sorted(temps.items()):
            hdd, cdd = degree_days(tavg_f, self.settings.base_temp_f)
            rows.append(
                {
                    "date": day,
                    "location": location.name,
                    "tavg_f": round(tavg_f, 2),
                    "hdd": hdd,
                    "cdd": cdd,
                    "is_forecast": day > self.today,
                }
            )
        return rows

    def run(self, start: date, end: date) -> int:
        if end >= self.today:
            end = self.today + timedelta(days=self.settings.forecast_days)

        with tracked_run(self.settings.db_url, self.name) as run:
            failed = []
            for location in self.settings.locations:
                # Keep going if one city fails so the rest still load
                try:
                    rows = self.location_rows(location, start, end)
                except (RuntimeError, KeyError) as exc:
                    log.error("weather %s failed: %s", location.name, exc)
                    failed.append(location.name)
                    continue
                run.rows += self.save(rows)
                log.info("weather %s: %d rows", location.name, len(rows))

            if failed:
                raise RuntimeError(f"weather failed for: {', '.join(failed)}")
        return run.rows

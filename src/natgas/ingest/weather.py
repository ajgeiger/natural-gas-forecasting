"""Daily temperatures and degree days from Open-Meteo (no API key needed).

History comes from the archive API, which lags a few days behind. The forecast
API fills the gap from the last archived day through the forecast horizon.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import requests

from natgas.config import Location, Settings
from natgas.db import WeatherDaily, session_scope, upsert
from natgas.ingest.runs import tracked_run

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The newest archive days are still being written and can fail to load
ARCHIVE_LAG_DAYS = 7
FORECAST_MAX_PAST_DAYS = 90
TIMEOUT_SECONDS = 60
# Long archive requests sometimes fail mid-stream, so history is pulled a year at a time
CHUNK_DAYS = 366
MAX_ATTEMPTS = 5


def degree_days(tavg_f: float, base_temp_f: float) -> tuple[float, float]:
    """Heating and cooling degree days for one day."""
    hdd = max(0.0, base_temp_f - tavg_f)
    cdd = max(0.0, tavg_f - base_temp_f)
    return round(hdd, 2), round(cdd, 2)


def date_chunks(start: date, end: date, chunk_days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=chunk_days - 1))
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def get_json(url: str, params: dict) -> dict:
    """GET with a few retries on server errors or a cut-off response."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            client_error = (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and 400 <= exc.response.status_code < 500
                and exc.response.status_code != 429
            )
            if client_error or attempt == MAX_ATTEMPTS:
                raise
            log.warning("open-meteo request failed (%s), retrying", exc)
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_daily_mean(url: str, location: Location, start: date, end: date) -> dict[date, float]:
    temps: dict[date, float] = {}
    for chunk_start, chunk_end in date_chunks(start, end):
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


def location_rows(
    location: Location, start: date, end: date, base_temp_f: float, today: date
) -> list[dict]:
    temps: dict[date, float] = {}

    archive_end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
    if start <= archive_end:
        temps.update(fetch_daily_mean(ARCHIVE_URL, location, start, archive_end))

    # The forecast API only reaches about three months back
    forecast_start = max(temps) + timedelta(days=1) if temps else start
    forecast_start = max(forecast_start, today - timedelta(days=FORECAST_MAX_PAST_DAYS))
    if forecast_start <= end:
        temps.update(fetch_daily_mean(FORECAST_URL, location, forecast_start, end))

    rows = []
    for day, tavg_f in sorted(temps.items()):
        hdd, cdd = degree_days(tavg_f, base_temp_f)
        rows.append(
            {
                "date": day,
                "location": location.name,
                "tavg_f": round(tavg_f, 2),
                "hdd": hdd,
                "cdd": cdd,
                "is_forecast": day > today,
            }
        )
    return rows


def ingest_weather(settings: Settings, start: date, end: date, today: date | None = None) -> int:
    """Load observed weather for [start, end], plus the forecast window when end is today."""
    today = today or date.today()
    if end >= today:
        end = today + timedelta(days=settings.forecast_days)

    with tracked_run(settings.db_url, "weather") as run:
        failed = []
        for location in settings.locations:
            # Keep going if one city fails so the rest still load
            try:
                rows = location_rows(location, start, end, settings.base_temp_f, today)
            except (requests.RequestException, ValueError, KeyError) as exc:
                log.error("weather %s failed: %s", location.name, exc)
                failed.append(location.name)
                continue
            with session_scope(settings.db_url) as session:
                run.rows += upsert(session, WeatherDaily, rows)
            log.info("weather %s: %d rows", location.name, len(rows))

        if failed:
            raise RuntimeError(f"weather failed for: {', '.join(failed)}")
    return run.rows

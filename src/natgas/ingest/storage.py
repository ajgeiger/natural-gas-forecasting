"""Weekly working gas in storage from the EIA API v2."""

from __future__ import annotations

import logging
from datetime import date

import requests

from natgas.config import Settings
from natgas.db import StorageWeekly, session_scope, upsert
from natgas.ingest.runs import tracked_run

log = logging.getLogger(__name__)

EIA_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
PAGE_SIZE = 5000
TIMEOUT_SECONDS = 60


def fetch_storage(api_key: str, series: str, start: date, end: date) -> list[dict]:
    """Return raw EIA records for one series, following pagination."""
    records: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": series,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        # Errors are re-raised without the request url so the api key stays out of logs
        try:
            response = requests.get(EIA_URL, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise RuntimeError(f"EIA request failed: {type(exc).__name__}") from None
        if not response.ok:
            raise RuntimeError(f"EIA request failed with status {response.status_code}")
        body = response.json()["response"]
        page = body.get("data", [])
        records.extend(page)

        offset += len(page)
        if not page or offset >= int(body.get("total", 0)):
            return records


def to_rows(records: list[dict], region: str) -> list[dict]:
    rows = []
    for record in records:
        if record.get("value") is None:
            continue
        rows.append(
            {
                "week_ending": date.fromisoformat(record["period"]),
                "region": region,
                "value_bcf": float(record["value"]),
            }
        )
    return rows


def ingest_storage(settings: Settings, start: date, end: date) -> int:
    with tracked_run(settings.db_url, "storage") as run:
        if not settings.eia_api_key:
            message = "EIA_API_KEY is not set, skipping storage"
            log.warning(message)
            run.skip(message)
            return 0

        records = fetch_storage(settings.eia_api_key, settings.storage_series, start, end)
        rows = to_rows(records, settings.storage_region)
        with session_scope(settings.db_url) as session:
            run.rows = upsert(session, StorageWeekly, rows)
    return run.rows

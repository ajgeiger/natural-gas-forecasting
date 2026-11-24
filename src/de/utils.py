"""Pieces shared by every data source: the Loader base class, run logging,
HTTP requests with retries, and date ranges."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta

import requests

from config import Settings
from data.database import IngestionRun, session_scope, upsert
from data.database.models import utc_now

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 5


class RunResult:
    """Filled in by the loader inside a tracked_run block."""

    def __init__(self) -> None:
        self.rows = 0
        self.status = "success"
        self.message: str | None = None

    def skip(self, message: str) -> None:
        self.status = "skipped"
        self.message = message


@contextmanager
def tracked_run(db_url: str, source: str) -> Iterator[RunResult]:
    """Log one loader run to the ingestion_runs table, including failures."""
    with session_scope(db_url) as session:
        run = IngestionRun(source=source, started_at=utc_now(), status="running")
        session.add(run)
        session.flush()
        run_id = run.id

    result = RunResult()
    try:
        yield result
    except Exception as exc:
        result.status = "failed"
        result.message = str(exc)[:1000]
        raise
    finally:
        with session_scope(db_url) as session:
            run = session.get(IngestionRun, run_id)
            run.finished_at = utc_now()
            run.rows = result.rows
            run.status = result.status
            run.message = result.message
        log.info("%s: %s, %d rows", source, result.status, result.rows)


class Loader:
    """Pulls rows from one source and upserts them into one table.

    Subclasses set name and table and implement rows().
    """

    name = ""
    table = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rows(self, start: date, end: date) -> list[dict]:
        raise NotImplementedError

    def run(self, start: date, end: date) -> int:
        with tracked_run(self.settings.db_url, self.name) as run:
            rows = self.rows(start, end)
            if not rows:
                log.warning("%s: no data returned for %s to %s", self.name, start, end)
            run.rows = self.save(rows)
        return run.rows

    def save(self, rows: list[dict]) -> int:
        with session_scope(self.settings.db_url) as session:
            return upsert(session, self.table, rows)


def get_json(url: str, params: dict) -> dict:
    """GET with retries on server errors or a cut-off response.

    Errors name only the base url, so keys passed in params stay out of logs.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            error = f"{url} failed: {type(exc).__name__}"
        else:
            status = response.status_code
            if response.ok:
                try:
                    return response.json()
                except ValueError:
                    error = f"{url} returned bad json"
            elif 400 <= status < 500 and status != 429:
                raise RuntimeError(f"{url} returned {status}")
            else:
                error = f"{url} returned {status}"

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(error)
        log.warning("%s, retrying", error)
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Split [start, end] into back to back ranges of at most chunk_days."""
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=chunk_days - 1))
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks

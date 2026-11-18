"""Shared run logic for the data loaders."""

from __future__ import annotations

import logging
from datetime import date

from natgas.config import Settings
from natgas.db import session_scope, upsert
from natgas.ingest.runs import tracked_run

log = logging.getLogger(__name__)


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

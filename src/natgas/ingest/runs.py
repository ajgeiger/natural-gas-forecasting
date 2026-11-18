"""Record each loader run in the ingestion_runs table."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from natgas.db import IngestionRun, session_scope
from natgas.db.models import utc_now

log = logging.getLogger(__name__)


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

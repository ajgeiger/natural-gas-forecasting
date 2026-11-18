from datetime import date

from sqlalchemy import select

from natgas.db import IngestionRun, StorageWeekly, session_scope
from natgas.ingest.base import Loader


class FakeLoader(Loader):
    name = "fake"
    table = StorageWeekly

    def rows(self, start, end):
        return [{"week_ending": start, "region": "lower48", "value_bcf": 100.0}]


def test_run_saves_rows_and_logs_run(settings):
    assert FakeLoader(settings).run(date(2024, 1, 5), date(2024, 1, 5)) == 1

    with session_scope(settings.db_url) as session:
        run = session.scalars(select(IngestionRun)).one()
        assert session.query(StorageWeekly).count() == 1
    assert run.source == "fake"
    assert run.status == "success"
    assert run.rows == 1

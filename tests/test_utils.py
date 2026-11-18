from datetime import date

import pytest
import responses
from sqlalchemy import select

from natgas.db import IngestionRun, StorageWeekly, session_scope
from natgas.de import utils

URL = "https://example.com/data"


class FakeLoader(utils.Loader):
    name = "fake"
    table = StorageWeekly

    def rows(self, start, end):
        return [{"week_ending": start, "region": "lower48", "value_bcf": 100.0}]


def test_loader_saves_rows_and_logs_run(settings):
    assert FakeLoader(settings).run(date(2024, 1, 5), date(2024, 1, 5)) == 1

    with session_scope(settings.db_url) as session:
        run = session.scalars(select(IngestionRun)).one()
        assert session.query(StorageWeekly).count() == 1
    assert run.source == "fake"
    assert run.status == "success"
    assert run.rows == 1


def test_tracked_run_records_failure(settings):
    with pytest.raises(ValueError), utils.tracked_run(settings.db_url, "prices"):
        raise ValueError("boom")

    with session_scope(settings.db_url) as session:
        run = session.scalars(select(IngestionRun)).one()
    assert run.status == "failed"
    assert run.message == "boom"
    assert run.finished_at is not None


@responses.activate
def test_get_json_retries_on_server_error(monkeypatch):
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    responses.get(URL, status=500)
    responses.get(URL, json={"ok": True})

    assert utils.get_json(URL, {"day": "2024-01-01"}) == {"ok": True}
    assert len(responses.calls) == 2


@responses.activate
def test_get_json_error_hides_params():
    responses.get(URL, status=403)

    with pytest.raises(RuntimeError) as error:
        utils.get_json(URL, {"api_key": "secret-key"})
    assert "secret-key" not in str(error.value)
    assert len(responses.calls) == 1


def test_date_chunks_cover_range():
    chunks = utils.date_chunks(date(2020, 1, 1), date(2020, 1, 10), chunk_days=4)
    assert chunks == [
        (date(2020, 1, 1), date(2020, 1, 4)),
        (date(2020, 1, 5), date(2020, 1, 8)),
        (date(2020, 1, 9), date(2020, 1, 10)),
    ]

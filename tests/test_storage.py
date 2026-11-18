from dataclasses import replace
from datetime import date

import pytest
import responses
from sqlalchemy import select

from natgas.db import IngestionRun, StorageWeekly, session_scope
from natgas.ingest import storage


def test_skips_without_api_key(settings):
    assert storage.StorageLoader(settings).run(date(2024, 1, 1), date(2024, 2, 1)) == 0

    with session_scope(settings.db_url) as session:
        run = session.scalars(select(IngestionRun)).one()
    assert run.status == "skipped"


@responses.activate
def test_ingest_storage_follows_pages(settings, monkeypatch):
    monkeypatch.setattr(storage, "PAGE_SIZE", 2)
    settings = replace(settings, eia_api_key="test-key")
    records = [
        {"period": "2024-01-05", "series": "NW2_EPG0_SWO_R48_BCF", "value": "3336"},
        {"period": "2024-01-12", "series": "NW2_EPG0_SWO_R48_BCF", "value": 3182},
        {"period": "2024-01-19", "series": "NW2_EPG0_SWO_R48_BCF", "value": None},
    ]
    responses.get(storage.EIA_URL, json={"response": {"total": 3, "data": records[:2]}})
    responses.get(storage.EIA_URL, json={"response": {"total": 3, "data": records[2:]}})

    assert storage.StorageLoader(settings).run(date(2024, 1, 1), date(2024, 1, 31)) == 2
    assert "offset=2" in responses.calls[1].request.url

    with session_scope(settings.db_url) as session:
        row = session.get(StorageWeekly, (date(2024, 1, 5), "lower48"))
    assert row.value_bcf == 3336.0


@responses.activate
def test_error_message_hides_api_key(settings):
    settings = replace(settings, eia_api_key="secret-key")
    responses.get(storage.EIA_URL, status=403)

    with pytest.raises(RuntimeError) as error:
        storage.StorageLoader(settings).run(date(2024, 1, 1), date(2024, 1, 31))
    assert "secret-key" not in str(error.value)

    with session_scope(settings.db_url) as session:
        run = session.scalars(select(IngestionRun)).one()
    assert "secret-key" not in run.message

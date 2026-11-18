from datetime import date

import pandas as pd

from natgas.db import Price, session_scope
from natgas.ingest import prices


def fake_history(*_args, **_kwargs) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04"], tz="America/New_York", name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [2.5, 2.6, 2.7],
            "High": [2.6, 2.7, None],
            "Low": [2.4, 2.5, 2.6],
            "Close": [2.55, 2.65, 2.75],
            "Volume": [100, 200, 300],
        },
        index=index,
    )


def test_to_rows_drops_missing_high():
    rows = prices.to_rows(fake_history())
    assert [row["date"] for row in rows] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert rows[0]["high"] == 2.6
    assert rows[0]["volume"] == 100


def test_ingest_prices(monkeypatch, settings):
    monkeypatch.setattr(prices, "fetch_prices", fake_history)
    count = prices.ingest_prices(settings, date(2024, 1, 1), date(2024, 1, 5))
    assert count == 2

    # A rerun should not add rows
    prices.ingest_prices(settings, date(2024, 1, 1), date(2024, 1, 5))
    with session_scope(settings.db_url) as session:
        assert session.query(Price).count() == 2

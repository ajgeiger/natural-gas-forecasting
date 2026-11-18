from datetime import date

import pandas as pd

from natgas.db import Price, session_scope
from natgas.de.prices import loader as prices


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


def test_to_rows_drops_missing_high(settings):
    rows = prices.PriceLoader(settings).to_rows(fake_history())
    assert [row["date"] for row in rows] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert rows[0]["high"] == 2.6
    assert rows[0]["volume"] == 100


def test_run_upserts_prices(monkeypatch, settings):
    monkeypatch.setattr(prices.PriceLoader, "fetch", fake_history)
    loader = prices.PriceLoader(settings)
    assert loader.run(date(2024, 1, 1), date(2024, 1, 5)) == 2

    # A rerun should not add rows
    loader.run(date(2024, 1, 1), date(2024, 1, 5))
    with session_scope(settings.db_url) as session:
        assert session.query(Price).count() == 2

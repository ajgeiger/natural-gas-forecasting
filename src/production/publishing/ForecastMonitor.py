"""Accuracy of matured live forecasts, as database views."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.database import get_engine
from data.database.SqlReader import SqlReader

ROLLING_AS_OF_DATES = 20
ALL_DAYS = "all days"
# Production mixes the event winners, so a stage can hold more than one model
MIXED_MODELS = "tournament"
MODEL_LABEL = f"CASE WHEN MIN(model) = MAX(model) THEN MIN(model) ELSE '{MIXED_MODELS}' END"

# Each forecast once in its own bucket and once in the all days bucket
SCORED = f"""
    SELECT model, stage, horizon_bucket, as_of_date, error, actual FROM forecast_actuals
    UNION ALL
    SELECT model, stage, '{ALL_DAYS}', as_of_date, error, actual FROM forecast_actuals
"""

ACCURACY_VIEW = f"""
CREATE VIEW forecast_accuracy AS
WITH scored AS ({SCORED})
SELECT
    {MODEL_LABEL} AS model,
    stage,
    horizon_bucket,
    COUNT(*) AS forecasts,
    COUNT(DISTINCT as_of_date) AS as_of_dates,
    MIN(as_of_date) AS first_as_of_date,
    MAX(as_of_date) AS last_as_of_date,
    AVG(ABS(error)) AS mae,
    100.0 * AVG(ABS(error) / ABS(actual)) AS mape,
    100.0 * SUM(ABS(error)) / SUM(ABS(actual)) AS wape,
    AVG(error) AS bias
FROM scored
GROUP BY stage, horizon_bucket
"""

ROLLING_VIEW = f"""
CREATE VIEW forecast_accuracy_rolling AS
WITH scored AS ({SCORED}),
by_date AS (
    SELECT
        {MODEL_LABEL} AS model,
        stage,
        horizon_bucket,
        as_of_date,
        COUNT(*) AS forecasts,
        SUM(ABS(error)) AS abs_error,
        SUM(ABS(error) / ABS(actual)) AS pct_error,
        SUM(ABS(actual)) AS actual_total,
        SUM(error) AS error_total
    FROM scored
    GROUP BY stage, horizon_bucket, as_of_date
),
rolling AS (
    SELECT
        model,
        stage,
        horizon_bucket,
        as_of_date,
        COUNT(*) OVER recent AS as_of_dates,
        SUM(forecasts) OVER recent AS forecasts,
        SUM(abs_error) OVER recent AS abs_error,
        SUM(pct_error) OVER recent AS pct_error,
        SUM(actual_total) OVER recent AS actual_total,
        SUM(error_total) OVER recent AS error_total
    FROM by_date
    WINDOW recent AS (
        PARTITION BY stage, horizon_bucket
        ORDER BY as_of_date
        ROWS BETWEEN {ROLLING_AS_OF_DATES - 1} PRECEDING AND CURRENT ROW
    )
)
SELECT
    model,
    stage,
    horizon_bucket,
    as_of_date,
    as_of_dates,
    forecasts,
    abs_error / forecasts AS mae,
    100.0 * pct_error / forecasts AS mape,
    100.0 * abs_error / actual_total AS wape,
    error_total / forecasts AS bias
FROM rolling
"""

METRICS = ["mae", "mape", "wape", "bias"]
ACCURACY_COLUMNS = [
    "model",
    "stage",
    "horizon_bucket",
    "forecasts",
    "as_of_dates",
    "first_as_of_date",
    "last_as_of_date",
    *METRICS,
]
ROLLING_COLUMNS = [
    "model",
    "stage",
    "horizon_bucket",
    "as_of_date",
    "as_of_dates",
    "forecasts",
    *METRICS,
]


class ForecastMonitor:
    """MAE, MAPE, WAPE, and bias of matured forecasts, by stage and horizon bucket.

    Every stage gets its rows: production, each contender such as
    chronos:champion or baseline:naive, and the legacy stages. A stage
    normally holds one model, shown in the model column. Production stitches
    the event winners together, so when its forecasts in a bucket come from
    more than one model the column says "tournament".

    forecast_accuracy scores every matured forecast. forecast_accuracy_rolling
    gives, for each as-of date, the scores over that date and the 19 before it.
    MAE and bias are in dollars per MMBtu, MAPE and WAPE in percent. Recent
    as-of dates count only their days that have matured so far.
    """

    def __init__(self, db_url: str) -> None:
        self.engine = get_engine(db_url)

    def create_views(self) -> None:
        """Drop and recreate both views so they always match this code."""
        with self.engine.begin() as connection:
            for name, sql in [
                ("forecast_accuracy", ACCURACY_VIEW),
                ("forecast_accuracy_rolling", ROLLING_VIEW),
            ]:
                connection.execute(text(f"DROP VIEW IF EXISTS {name}"))
                connection.execute(text(sql))

    def accuracy(self) -> pd.DataFrame:
        query = text(f"SELECT {', '.join(ACCURACY_COLUMNS)} FROM forecast_accuracy")
        return SqlReader.read(self.engine, query, ACCURACY_COLUMNS)

    def rolling(self) -> pd.DataFrame:
        query = text(
            f"SELECT {', '.join(ROLLING_COLUMNS)} FROM forecast_accuracy_rolling "
            "ORDER BY stage, horizon_bucket, as_of_date"
        )
        return SqlReader.read(self.engine, query, ROLLING_COLUMNS)

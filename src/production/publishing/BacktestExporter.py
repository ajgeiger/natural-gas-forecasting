"""Writes the backtest forecasts of the best config per model for the dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert

from backtesting.Backtester import Backtester
from data.database import BacktestForecast, session_scope
from data.provider.ForecastDataProvider import ForecastDataProvider
from models.ModelCatalog import ModelCatalog
from production.registry.GridSearchRuns import GridSearchRuns

log = logging.getLogger(__name__)

COLUMNS = [
    "as_of_date",
    "target_date",
    "horizon_day",
    "model",
    "model_version",
    "forecast",
    "lower",
    "upper",
    "actual",
]
# A rerun must land this close to the logged forecasts to be trusted
MATCH_TOLERANCE = 1e-4


class BacktestExporter:
    """One tidy table of every backtest forecast from the latest grid search.

    The forecasts are the ones the grid search logged to MLflow, not new
    results. The grid search only kept the Chronos-2 median, so Chronos-2 is
    backtested again on the same as-of dates to get its 10th and 90th
    percentiles. The interval is kept only where the rerun median matches the
    logged one. Other models have no interval, so lower and upper are empty.

    model_version is the registry version published from this search, or the
    MLflow run id of the config when it was never registered.
    """

    def __init__(
        self,
        runs: GridSearchRuns,
        provider: ForecastDataProvider,
        db_url: str,
        registered_model_name: str,
        n_jobs: int = 1,
    ) -> None:
        self.runs = runs
        self.provider = provider
        self.db_url = db_url
        self.registered_model_name = registered_model_name
        self.n_jobs = n_jobs

    def build(self) -> pd.DataFrame:
        search = self.runs.latest_search()
        versions = self.runs.registered_versions(search, self.registered_model_name)
        frames = []
        for model_name, run in sorted(self.runs.best_runs(search).items()):
            forecasts = self.runs.forecasts(run)
            makes_intervals = ModelCatalog.model_class(model_name).makes_intervals
            if makes_intervals and "lower" not in forecasts:
                forecasts = self.add_intervals(forecasts, run)
            version = versions.get(model_name, f"run-{run.info.run_id}")
            frames.append(self.tidy(forecasts, model_name, version))
            log.info("%s %s: %d forecasts", model_name, version, len(forecasts))
        return pd.concat(frames, ignore_index=True)

    def add_intervals(self, forecasts: pd.DataFrame, run) -> pd.DataFrame:
        """Add lower and upper from a rerun, for as-of dates where the rerun matches.

        Revised source data, such as recent weather the archive corrected
        after the search, can move a rerun. Those as-of dates keep their logged
        forecasts and get no interval.
        """
        config = self.runs.config_for(run)
        as_of_dates = sorted(pd.Timestamp(day) for day in forecasts["as_of"].unique())
        rerun = Backtester(self.provider, as_of_dates, self.n_jobs).run(config).forecasts()
        merged = forecasts.merge(
            rerun[["as_of", "date", "prediction", "lower", "upper"]],
            on=["as_of", "date"],
            how="left",
            suffixes=("", "_rerun"),
        )
        gap = (merged["prediction"] - merged["prediction_rerun"]).abs()
        changed = gap.groupby(merged["as_of"]).transform("max") > MATCH_TOLERANCE
        merged.loc[changed, ["lower", "upper"]] = float("nan")
        if changed.any():
            dates = sorted(merged.loc[changed, "as_of"].dt.date.unique())
            log.warning("%s rerun differs on %s, no interval there", config.model_name, dates)
        return merged.drop(columns="prediction_rerun")

    @staticmethod
    def tidy(forecasts: pd.DataFrame, model_name: str, version: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "as_of_date": forecasts["as_of"].dt.date,
                "target_date": forecasts["date"].dt.date,
                "horizon_day": forecasts["step"].astype(int),
                "model": model_name,
                "model_version": version,
                "forecast": forecasts["prediction"],
                "lower": forecasts.get("lower", float("nan")),
                "upper": forecasts.get("upper", float("nan")),
                "actual": forecasts["actual"],
            }
        )
        frame[["forecast", "lower", "upper"]] = frame[["forecast", "lower", "upper"]].astype(float)
        return frame[COLUMNS]

    def write(self, frame: pd.DataFrame, parquet_path: str | Path) -> int:
        """Replace the backtest_forecasts table and write the same rows to Parquet."""
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
        with session_scope(self.db_url) as session:
            session.execute(delete(BacktestForecast))
            session.execute(insert(BacktestForecast), rows)
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, index=False)
        return len(rows)

"""All backtests for one model config."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.BacktestResult import BacktestResult
from backtesting.ForecastMetrics import ForecastMetrics
from models.config.AbstractConfigWrapper import AbstractConfigWrapper

# Days ahead grouped into short, medium, and long range
HORIZON_BUCKETS = [(1, 5), (6, 15), (16, 30)]


@dataclass
class BacktestReport:
    """Collects BacktestResults for one config and scores them together."""

    config: AbstractConfigWrapper
    results: list[BacktestResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def forecasts(self) -> pd.DataFrame:
        """Every forecast from every backtest in one table."""
        return pd.concat([result.to_frame() for result in self.results], ignore_index=True)

    def metrics(self) -> dict[str, float]:
        """Scores over all forecasts pooled together."""
        table = self.forecasts()
        return ForecastMetrics.score(table["actual"], table["prediction"])

    def metrics_by_step(self) -> pd.DataFrame:
        """Scores for 1 day ahead, 2 days ahead, and so on."""
        table = self.forecasts()
        rows = {
            step: ForecastMetrics.score(group["actual"], group["prediction"])
            for step, group in table.groupby("step")
        }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("step")

    def metrics_by_bucket(self, buckets: list[tuple[int, int]] = HORIZON_BUCKETS) -> pd.DataFrame:
        """Scores for ranges of days ahead, for example days 1 to 5.

        Buckets that start past the longest forecast are skipped.
        """
        table = self.forecasts()
        rows = {}
        for first, last in buckets:
            group = table[table["step"].between(first, last)]
            if len(group):
                rows[f"days {first} to {last}"] = ForecastMetrics.score(
                    group["actual"], group["prediction"]
                )
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("days_ahead")

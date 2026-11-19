"""All backtests for one model config."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from natgas.forecasting.backtesting.BacktestResult import BacktestResult
from natgas.forecasting.backtesting.ForecastMetrics import ForecastMetrics
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper


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

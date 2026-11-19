"""Runs rolling-origin backtests for any model config."""

from __future__ import annotations

import logging
import time

import pandas as pd
from joblib import Parallel, delayed

from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.backtesting.BacktestResult import BacktestResult
from natgas.forecasting.backtesting.ForecastMetrics import ForecastMetrics
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper
from natgas.forecasting.models.ModelCatalog import ModelCatalog

log = logging.getLogger(__name__)


class Backtester:
    """Replays history as if the model were run on each as-of date.

    For every as-of date it asks the data provider for that window, checks for
    leakage, builds a fresh model from the config, trains it, predicts the test
    dates, and scores the result. Before any of that it checks the config
    against the provider: the features it needs, its longest horizon, and the
    history it needs. n_jobs runs as-of dates in parallel processes.
    """

    def __init__(
        self, provider: ForecastDataProvider, as_of_dates: list[pd.Timestamp], n_jobs: int = 1
    ) -> None:
        if not as_of_dates:
            raise ValueError("Backtester needs at least one as-of date")
        self.provider = provider
        self.as_of_dates = list(as_of_dates)
        self.n_jobs = n_jobs

    def check_config(self, config: AbstractConfigWrapper) -> None:
        """Make sure the provider can give this model what it needs."""
        missing = set(config.required_features) - set(self.provider.feature_columns)
        if missing:
            raise ValueError(f"{config.model_name} needs features the provider lacks: {missing}")
        limit = config.max_horizon_days
        if limit is not None and self.provider.horizon_days > limit:
            raise ValueError(f"{config.model_name} forecasts at most {limit} days ahead")
        first_rows = len(self.provider.target.loc[: self.as_of_dates[0]])
        if first_rows < config.min_training_rows:
            raise ValueError(
                f"{config.model_name} needs {config.min_training_rows} training rows, "
                f"the first as-of date has {first_rows}"
            )

    def run_one(self, config: AbstractConfigWrapper, as_of: pd.Timestamp) -> BacktestResult:
        """Train and score one model config on one as-of date."""
        window = self.provider.window(as_of)
        self.check_no_leakage(window.as_of, window.training_data, window.forecast_dates)

        model = ModelCatalog.build(config)
        model.train(window.training_data)
        prediction = model.predict(window.future_features)

        forecasts = pd.DataFrame(
            {
                "step": range(1, len(prediction) + 1),
                "actual": window.test_target,
                "prediction": prediction,
            }
        )
        return BacktestResult(
            as_of=window.as_of,
            training_rows=len(window.training_data),
            last_training_date=window.training_data.index.max(),
            forecasts=forecasts,
            metrics=ForecastMetrics.score(forecasts["actual"], forecasts["prediction"]),
        )

    def run(self, config: AbstractConfigWrapper) -> BacktestReport:
        """Backtest one config on every as-of date."""
        self.check_config(config)
        started = time.perf_counter()
        if self.n_jobs == 1:
            results = [self.run_one(config, as_of) for as_of in self.as_of_dates]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_one)(config, as_of) for as_of in self.as_of_dates
            )
        report = BacktestReport(config, results, time.perf_counter() - started)
        log.info(
            "%s %s: %d backtests, mae %.4f, %.0fs",
            config.model_name,
            config.to_json(),
            len(results),
            report.metrics()["mae"],
            report.seconds,
        )
        return report

    @staticmethod
    def check_no_leakage(
        as_of: pd.Timestamp, training_data: pd.DataFrame, forecast_dates: pd.DatetimeIndex
    ) -> None:
        """Stop the run if any training row is after the as-of date."""
        if training_data.index.max() > as_of:
            raise AssertionError(f"Training data goes past the as-of date {as_of.date()}")
        if forecast_dates.min() <= as_of:
            raise AssertionError(f"Forecast dates must start after {as_of.date()}")

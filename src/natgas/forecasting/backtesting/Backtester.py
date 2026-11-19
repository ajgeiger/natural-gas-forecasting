"""Runs rolling-origin backtests for any model."""

from __future__ import annotations

import logging

import pandas as pd

from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.backtesting.BacktestResult import BacktestResult
from natgas.forecasting.backtesting.ForecastMetrics import ForecastMetrics
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.models.AbstractModelWrapper import AbstractModelWrapper

log = logging.getLogger(__name__)


class Backtester:
    """Replays history as if the model were run on each as-of date.

    For every as-of date it asks the data provider for that window, trains a
    fresh model on the training data, predicts the test dates, and scores the
    result. A leakage check runs on every window before training.
    """

    def __init__(self, provider: ForecastDataProvider, as_of_dates: list[pd.Timestamp]) -> None:
        if not as_of_dates:
            raise ValueError("Backtester needs at least one as-of date")
        self.provider = provider
        self.as_of_dates = list(as_of_dates)

    def run_one(
        self, model_class: type[AbstractModelWrapper], hyperparameters: dict, as_of: pd.Timestamp
    ) -> BacktestResult:
        """Train and score one model on one as-of date."""
        window = self.provider.window(as_of)
        self.check_no_leakage(window.as_of, window.training_data, window.forecast_dates)

        model = model_class(hyperparameters)
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

    def run(
        self, model_class: type[AbstractModelWrapper], hyperparameters: dict | None = None
    ) -> BacktestReport:
        """Backtest one model and one set of hyperparameters on every as-of date."""
        hyperparameters = hyperparameters or {}
        report = BacktestReport(model_class.name, hyperparameters)
        for as_of in self.as_of_dates:
            report.results.append(self.run_one(model_class, hyperparameters, as_of))
        log.info(
            "%s %s: %d backtests, mae %.4f",
            model_class.name,
            hyperparameters,
            len(report.results),
            report.metrics()["mae"],
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

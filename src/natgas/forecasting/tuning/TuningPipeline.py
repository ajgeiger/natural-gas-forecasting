"""End to end: tune, backtest, log to MLflow, and publish the best model."""

from __future__ import annotations

import logging

import mlflow

from natgas.config import Settings
from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.tuning.HyperparameterTuner import HyperparameterTuner
from natgas.forecasting.tuning.ModelPublisher import ModelPublisher
from natgas.forecasting.tuning.TuningConfig import TuningConfig
from natgas.forecasting.tuning.TuningOutcome import TuningOutcome

log = logging.getLogger(__name__)


class TuningPipeline:
    """Runs the full workflow from a TuningConfig.

    1. Build the data provider and the rolling as-of dates
    2. Backtest every hyperparameter config for every model
    3. Log each config and the summary table to an MLflow experiment
    4. Retrain the best config on all data and register it in the model registry
    """

    def __init__(self, config: TuningConfig, provider: ForecastDataProvider) -> None:
        self.config = config
        self.provider = provider

    @classmethod
    def from_settings(cls, config: TuningConfig, settings: Settings) -> TuningPipeline:
        provider = ForecastDataProvider.from_settings(
            settings, config.horizon_days, config.target_column
        )
        return cls(config, provider)

    def run(self, publish: bool = True) -> TuningOutcome:
        as_of_dates = self.provider.backtest_dates(
            self.config.backtest_start, self.config.backtest_end, self.config.step_days
        )
        log.info(
            "%d as-of dates from %s to %s",
            len(as_of_dates),
            as_of_dates[0].date(),
            as_of_dates[-1].date(),
        )
        tuner = HyperparameterTuner(
            Backtester(self.provider, as_of_dates), self.config.selection_metric
        )

        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name="hyperparameter-search") as parent:
            mlflow.log_params(
                {
                    "horizon_days": self.config.horizon_days,
                    "step_days": self.config.step_days,
                    "backtest_count": len(as_of_dates),
                    "selection_metric": self.config.selection_metric,
                    "target_column": self.config.target_column,
                    "feature_columns": ",".join(self.provider.feature_columns),
                }
            )
            summary = tuner.run(self.config.model_grids)
            parent_run_id = parent.info.run_id

        best = tuner.best()
        version = None
        if publish:
            publisher = ModelPublisher(
                self.provider, self.config.registered_model_name, self.config.model_alias
            )
            with mlflow.start_run(run_name=f"publish-{best.model_name}"):
                mlflow.set_tag("search_run_id", parent_run_id)
                version = publisher.publish(best)

        return TuningOutcome(
            summary=summary,
            best_model_name=best.model_name,
            best_hyperparameters=best.hyperparameters,
            best_metrics=best.metrics(),
            registered_model_name=self.config.registered_model_name,
            registered_version=version,
            parent_run_id=parent_run_id,
        )

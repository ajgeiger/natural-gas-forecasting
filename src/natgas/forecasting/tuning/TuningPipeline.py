"""End to end: search, backtest, log to MLflow, and publish."""

from __future__ import annotations

import logging

import mlflow

from natgas.config import Settings
from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.optimization.AbstractOptimizer import AbstractOptimizer
from natgas.forecasting.optimization.OptimizerCatalog import OptimizerCatalog
from natgas.forecasting.tuning.ModelPublisher import ModelPublisher
from natgas.forecasting.tuning.TuningConfig import TuningConfig
from natgas.forecasting.tuning.TuningOutcome import TuningOutcome

log = logging.getLogger(__name__)


class TuningPipeline:
    """Runs the full workflow from a TuningConfig.

    1. Build the rolling as-of dates from the data provider
    2. Search each model's configs with grid or Bayesian search
    3. Log every config and the results table to an MLflow experiment
    4. Publish the best config if it beats naive by the margin, otherwise naive
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

    def build_backtester(self) -> Backtester:
        as_of_dates = self.provider.backtest_dates(
            self.config.backtest_start, self.config.backtest_end, self.config.step_days
        )
        log.info(
            "%d as-of dates from %s to %s",
            len(as_of_dates),
            as_of_dates[0].date(),
            as_of_dates[-1].date(),
        )
        return Backtester(self.provider, as_of_dates, self.config.n_jobs)

    def build_optimizer(self, backtester: Backtester) -> AbstractOptimizer:
        """Grid or Bayesian search, picked by search.optimizer in tuning.yaml."""
        return OptimizerCatalog.build(
            self.config.optimizer,
            backtester,
            self.config.selection_metric,
            self.config.n_trials,
            self.config.seed,
        )

    def build_publisher(self) -> ModelPublisher:
        return ModelPublisher(
            self.provider,
            self.config.registered_model_name,
            self.config.model_alias,
            self.config.selection_metric,
            self.config.min_improvement,
        )

    def run(self, publish: bool = True) -> TuningOutcome:
        backtester = self.build_backtester()
        optimizer = self.build_optimizer(backtester)

        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=f"{optimizer.method_name}-search") as parent:
            mlflow.log_params(
                {
                    "optimizer": optimizer.method_name,
                    "horizon_days": self.config.horizon_days,
                    "step_days": self.config.step_days,
                    "backtest_count": len(backtester.as_of_dates),
                    "selection_metric": self.config.selection_metric,
                    "target_column": self.config.target_column,
                    "feature_columns": ",".join(self.provider.feature_columns),
                }
            )
            summary = optimizer.run(self.config.model_settings)
            parent_run_id = parent.info.run_id

        publisher = self.build_publisher()
        decision = publisher.choose(optimizer.reports)
        log.info("publish decision: %s", decision.reason)
        version = None
        if publish:
            with mlflow.start_run(run_name=f"publish-{decision.report.model_name}"):
                mlflow.set_tags({"search_run_id": parent_run_id, "method": optimizer.method_name})
                version = publisher.publish(decision)

        return TuningOutcome(
            summary=summary,
            best_by_model=optimizer.best_by_model(),
            decision=decision,
            registered_model_name=self.config.registered_model_name,
            registered_version=version,
            parent_run_id=parent_run_id,
        )

"""Shared logic for every hyperparameter search method."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import mlflow
import pandas as pd

from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper
from natgas.forecasting.models.ModelCatalog import ModelCatalog

log = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    """Searches each model's settings by backtesting candidate configs.

    Subclasses only decide which configs to try next (search). This class does
    the rest: runs the backtester for each candidate, logs it as a nested MLflow
    run, skips configs it has already scored, and builds the results table.
    """

    method_name: ClassVar[str] = ""
    space_key: ClassVar[str] = ""

    def __init__(self, backtester: Backtester, selection_metric: str = "mae") -> None:
        self.backtester = backtester
        self.selection_metric = selection_metric
        self.reports: list[BacktestReport] = []
        self.scored: dict[str, BacktestReport] = {}
        self.failed: dict[str, str] = {}
        self.attempts = 0

    @abstractmethod
    def default_space(self, config_class: type[AbstractConfigWrapper]) -> dict | list[dict]:
        """The config class's own grid or search space."""

    @abstractmethod
    def search(self, config_class: type[AbstractConfigWrapper], space: dict | list[dict]) -> None:
        """Pick configs from the space and score each one with evaluate()."""

    def space_for(
        self, config_class: type[AbstractConfigWrapper], settings: dict
    ) -> dict | list[dict]:
        """Space from tuning.yaml if given, otherwise the config class default."""
        space = settings.get(self.space_key)
        return space if space is not None else self.default_space(config_class)

    def run(self, model_settings: dict[str, dict]) -> pd.DataFrame:
        """Search every model. Call inside an active MLflow run."""
        for model_name, settings in model_settings.items():
            config_class = ModelCatalog.config_class(model_name)
            self.search(config_class, self.space_for(config_class, settings or {}))
        summary = self.summary()
        self.log_summary(summary)
        return summary

    def evaluate(self, config: AbstractConfigWrapper) -> float:
        """Backtest one config, log it to MLflow, and return its score."""
        key = f"{config.model_name}:{config.to_json()}"
        if key in self.failed:
            return math.inf
        if key not in self.scored:
            self.attempts += 1
            with mlflow.start_run(run_name=f"{config.model_name}-{self.attempts}", nested=True):
                try:
                    report = self.backtester.run(config)
                except Exception as error:
                    # One bad config should not stop the whole search
                    log.warning("%s %s failed: %s", config.model_name, config.to_json(), error)
                    mlflow.set_tags({"model_name": config.model_name, "status": "failed"})
                    mlflow.log_params(config.to_mlflow_params())
                    mlflow.set_tag("error", str(error)[:500])
                    self.failed[key] = str(error)
                    return math.inf
                self.log_report(report)
            self.scored[key] = report
            self.reports.append(report)
        return self.scored[key].metrics()[self.selection_metric]

    def log_report(self, report: BacktestReport) -> None:
        mlflow.set_tags(
            {"model_name": report.model_name, "stage": "backtest", "method": self.method_name}
        )
        mlflow.log_params({"model_name": report.model_name, **report.config.to_mlflow_params()})
        mlflow.log_params(
            {
                "backtest_count": len(report.results),
                "first_as_of": report.results[0].as_of.date(),
                "last_as_of": report.results[-1].as_of.date(),
            }
        )
        mlflow.log_metrics({**report.metrics(), "backtest_seconds": report.seconds})
        for step, scores in report.metrics_by_step().iterrows():
            mlflow.log_metrics({f"{name}_step_{step}": value for name, value in scores.items()})
        for bucket, scores in report.metrics_by_bucket().iterrows():
            label = bucket.replace(" ", "_")
            mlflow.log_metrics({f"{name}_{label}": value for name, value in scores.items()})

        with TemporaryDirectory() as folder:
            forecasts_path = Path(folder) / "backtest_forecasts.csv"
            report.forecasts().to_csv(forecasts_path, index=False)
            mlflow.log_artifact(str(forecasts_path))

    def summary(self) -> pd.DataFrame:
        """One row per config, best first, with a comparison to the naive baseline."""
        rows = [
            {
                "method": self.method_name,
                "model_name": report.model_name,
                "config": report.config.to_json(),
                "backtests": len(report.results),
                "seconds": round(report.seconds, 1),
                **report.metrics(),
            }
            for report in self.reports
        ]
        table = pd.DataFrame(rows).sort_values(self.selection_metric, ignore_index=True)
        baseline = table.loc[table["model_name"] == "naive", self.selection_metric]
        if not baseline.empty:
            # Above zero means the config beats repeating the last price
            table["skill_vs_naive"] = 1 - table[self.selection_metric] / baseline.min()
        return table

    def log_summary(self, summary: pd.DataFrame) -> None:
        best = summary.iloc[0]
        mlflow.log_metric(f"best_{self.selection_metric}", float(best[self.selection_metric]))
        mlflow.set_tags(
            {
                "method": self.method_name,
                "best_model_name": best["model_name"],
                "best_config": best["config"],
                "configs_tried": len(summary),
            }
        )
        with TemporaryDirectory() as folder:
            summary_path = Path(folder) / "tuning_summary.csv"
            summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))

    def best_by_model(self) -> dict[str, BacktestReport]:
        """The best report for each model name."""
        best: dict[str, BacktestReport] = {}
        for report in self.reports:
            current = best.get(report.model_name)
            score = report.metrics()[self.selection_metric]
            if current is None or score < current.metrics()[self.selection_metric]:
                best[report.model_name] = report
        return best

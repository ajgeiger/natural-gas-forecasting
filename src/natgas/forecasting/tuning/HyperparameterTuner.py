"""Backtests every hyperparameter config and logs each one to MLflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.tuning.HyperparameterGrid import HyperparameterGrid

log = logging.getLogger(__name__)


class HyperparameterTuner:
    """Runs the backtester once per config and keeps a results table.

    Each config gets its own nested MLflow run with its hyperparameters, its
    pooled scores, scores by forecast step, and every individual forecast.
    The parent run holds the summary table for the whole search.
    """

    def __init__(self, backtester: Backtester, selection_metric: str = "mae") -> None:
        self.backtester = backtester
        self.selection_metric = selection_metric
        self.reports: list[BacktestReport] = []

    def run(self, model_grids: dict[str, dict]) -> pd.DataFrame:
        """Try every config for every model. Call inside an active MLflow run."""
        for model_name, grid in model_grids.items():
            model_class = ModelCatalog.get(model_name)
            for number, hyperparameters in enumerate(HyperparameterGrid(grid).configs(), 1):
                with mlflow.start_run(run_name=f"{model_name}-{number}", nested=True):
                    report = self.backtester.run(model_class, hyperparameters)
                    self.log_report(report)
                    self.reports.append(report)

        summary = self.summary()
        self.log_summary(summary)
        return summary

    def log_report(self, report: BacktestReport) -> None:
        mlflow.set_tags({"model_name": report.model_name, "stage": "backtest"})
        mlflow.log_params({"model_name": report.model_name, **report.hyperparameters})
        mlflow.log_params(
            {
                "backtest_count": len(report.results),
                "first_as_of": report.results[0].as_of.date(),
                "last_as_of": report.results[-1].as_of.date(),
            }
        )
        mlflow.log_metrics(report.metrics())
        for step, scores in report.metrics_by_step().iterrows():
            mlflow.log_metrics({f"{name}_step_{step}": value for name, value in scores.items()})

        with TemporaryDirectory() as folder:
            forecasts_path = Path(folder) / "backtest_forecasts.csv"
            report.forecasts().to_csv(forecasts_path, index=False)
            mlflow.log_artifact(str(forecasts_path))

    def summary(self) -> pd.DataFrame:
        """One row per config, best first, with a comparison to the naive baseline."""
        rows = []
        for report in self.reports:
            rows.append(
                {
                    "model_name": report.model_name,
                    "hyperparameters": json.dumps(report.hyperparameters, sort_keys=True),
                    "backtests": len(report.results),
                    **report.metrics(),
                }
            )
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
                "best_model_name": best["model_name"],
                "best_hyperparameters": best["hyperparameters"],
                "configs_tried": len(summary),
            }
        )
        with TemporaryDirectory() as folder:
            summary_path = Path(folder) / "tuning_summary.csv"
            summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))

    def best(self) -> BacktestReport:
        """The report with the lowest selection metric."""
        if not self.reports:
            raise RuntimeError("Run the tuner before asking for the best config")
        return min(self.reports, key=lambda report: report.metrics()[self.selection_metric])

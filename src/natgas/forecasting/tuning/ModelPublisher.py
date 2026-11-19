"""Picks the model to publish and registers it in MLflow."""

from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

import natgas
from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.tuning.MlflowModelAdapter import MODEL_FOLDER_KEY, MlflowModelAdapter
from natgas.forecasting.tuning.PublishDecision import PublishDecision

BASELINE_MODEL = "naive"

# Packages the published model needs, pinned to the versions it was trained with
MODEL_PACKAGES = [
    "mlflow",
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "greykite",
    "chronos-forecasting",
    "torch",
]


class ModelPublisher:
    """Publishes a new version in the MLflow model registry.

    choose() applies the publishing rule: the best config only wins if it beats
    the naive baseline by at least min_improvement (0.01 means 1 percent lower
    error). Otherwise the naive model is published, because a model that does
    not clearly beat "repeat the last price" is not worth the extra risk.

    publish() retrains the chosen config on every row up to the latest date,
    logs it as an MLflow pyfunc model with the package code bundled in,
    registers it, and points an alias such as "champion" at the new version.
    """

    def __init__(
        self,
        provider: ForecastDataProvider,
        registered_model_name: str,
        alias: str = "champion",
        selection_metric: str = "mae",
        min_improvement: float = 0.01,
    ) -> None:
        self.provider = provider
        self.registered_model_name = registered_model_name
        self.alias = alias
        self.selection_metric = selection_metric
        self.min_improvement = min_improvement

    def choose(self, reports: list[BacktestReport]) -> PublishDecision:
        """Best config if it clears the margin over naive, otherwise naive."""
        metric = self.selection_metric
        best = min(reports, key=lambda report: report.metrics()[metric])
        baselines = [report for report in reports if report.model_name == BASELINE_MODEL]
        if not baselines:
            return PublishDecision(best, "best config, no naive baseline to compare", None)

        baseline = min(baselines, key=lambda report: report.metrics()[metric])
        improvement = 1 - best.metrics()[metric] / baseline.metrics()[metric]
        if best.model_name == BASELINE_MODEL:
            return PublishDecision(baseline, "naive had the lowest error", 0.0)
        if improvement >= self.min_improvement:
            reason = f"{best.model_name} beat naive by {improvement:.2%}"
            return PublishDecision(best, reason, improvement)
        reason = (
            f"best was {best.model_name} at {improvement:.2%} better than naive, "
            f"below the {self.min_improvement:.2%} margin, so naive is kept"
        )
        return PublishDecision(baseline, reason, improvement)

    def publish(self, decision: PublishDecision, as_of: pd.Timestamp | None = None) -> str:
        """Train, log, and register the chosen model. Returns the new version number."""
        report = decision.report
        as_of = pd.Timestamp(as_of) if as_of is not None else self.provider.last_date
        window = self.provider.window(as_of)
        model = ModelCatalog.build(report.config)
        model.train(window.training_data)

        example_input = window.future_features.reset_index(names="date")
        example_output = (
            model.predict(window.future_features).rename_axis("date").reset_index(name="prediction")
        )
        trained_through = window.training_data.index.max().date()

        with TemporaryDirectory() as folder:
            model_folder = Path(folder) / "model"
            model.save(model_folder)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=MlflowModelAdapter(),
                artifacts={MODEL_FOLDER_KEY: str(model_folder)},
                code_paths=[str(Path(natgas.__file__).parent)],
                signature=infer_signature(example_input, example_output),
                input_example=example_input,
                pip_requirements=[f"{name}=={package_version(name)}" for name in MODEL_PACKAGES],
                registered_model_name=self.registered_model_name,
            )

        mlflow.set_tags(
            {"model_name": report.model_name, "stage": "publish", "reason": decision.reason}
        )
        mlflow.log_params({"model_name": report.model_name, **report.config.to_mlflow_params()})
        mlflow.log_params(
            {"trained_through": trained_through, "min_improvement": self.min_improvement}
        )
        mlflow.log_metrics({f"backtest_{name}": value for name, value in report.metrics().items()})

        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(self.registered_model_name, self.alias, version)
        client.update_model_version(
            self.registered_model_name,
            version,
            description=(
                f"{report.model_name} {report.config.to_json()}, trained through "
                f"{trained_through}. {decision.reason}."
            ),
        )
        for name, value in report.metrics().items():
            client.set_model_version_tag(
                self.registered_model_name, version, f"backtest_{name}", f"{value:.4f}"
            )
        return version

"""Picks the model to publish and registers it in MLflow."""

from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from backtesting.BacktestReport import BacktestReport
from config import SOURCE_PACKAGES, SOURCE_ROOT
from data.provider.ForecastDataProvider import ForecastDataProvider
from models.config.AbstractConfigWrapper import AbstractConfigWrapper
from models.ModelCatalog import ModelCatalog
from production.registry.MlflowModelAdapter import MODEL_FOLDER_KEY, MlflowModelAdapter
from production.registry.PublishDecision import PublishDecision

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
    publish_config() does the same for a config whose backtest scores are
    already known, which is how the tournament seeds each family registry.
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

    @staticmethod
    def report_metrics(report: BacktestReport) -> dict[str, float]:
        """Overall scores plus scores per horizon bucket, named like mae_days_1_to_5."""
        metrics = dict(report.metrics())
        for bucket, scores in report.metrics_by_bucket().iterrows():
            for name, value in scores.items():
                metrics[f"{name}_{bucket.replace(' ', '_')}"] = float(value)
        return metrics

    @staticmethod
    def code_paths() -> list[str]:
        """Every package and module under src, so MLflow can load the model on its own."""
        paths = []
        for name in SOURCE_PACKAGES:
            folder = SOURCE_ROOT / name
            paths.append(str(folder if folder.is_dir() else folder.with_suffix(".py")))
        return paths

    def publish(self, decision: PublishDecision, as_of: pd.Timestamp | None = None) -> str:
        """Train, log, and register the chosen model. Returns the new version number."""
        report = decision.report
        metrics = self.report_metrics(report)
        return self.publish_config(report.config, metrics, decision.reason, as_of)

    def publish_config(
        self,
        config: AbstractConfigWrapper,
        metrics: dict[str, float],
        reason: str,
        as_of: pd.Timestamp | None = None,
    ) -> str:
        """Train a config on data through as_of, log it, register it, and set the alias.

        metrics are its backtest scores. They are logged to the run and saved
        as backtest_ tags on the version, so the tournament can read the
        backtest MAE of each horizon bucket without running anything again.
        """
        as_of = pd.Timestamp(as_of) if as_of is not None else self.provider.last_date
        window = self.provider.window(as_of)
        model = ModelCatalog.build(config)
        model.train(window.training_data)

        features = window.future_features[config.required_features]
        example_input = features.reset_index(names="date")
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
                code_paths=self.code_paths(),
                signature=infer_signature(example_input, example_output),
                input_example=example_input,
                pip_requirements=[f"{name}=={package_version(name)}" for name in MODEL_PACKAGES],
                registered_model_name=self.registered_model_name,
            )

        mlflow.set_tags({"model_name": config.model_name, "stage": "publish", "reason": reason})
        mlflow.log_params({"model_name": config.model_name, **config.to_mlflow_params()})
        mlflow.log_params(
            {"trained_through": trained_through, "min_improvement": self.min_improvement}
        )
        mlflow.log_metrics({f"backtest_{name}": value for name, value in metrics.items()})

        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(self.registered_model_name, self.alias, version)
        client.update_model_version(
            self.registered_model_name,
            version,
            description=(
                f"{config.model_name} {config.to_json()}, trained through "
                f"{trained_through}. {reason}."
            ),
        )
        for name, value in metrics.items():
            client.set_model_version_tag(
                self.registered_model_name, version, f"backtest_{name}", f"{value:.6f}"
            )
        return version

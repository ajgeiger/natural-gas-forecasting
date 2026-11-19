"""Trains the winning config on all data and registers it in MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

import natgas
from natgas.forecasting.backtesting.BacktestReport import BacktestReport
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.tuning.MlflowModelAdapter import MODEL_FOLDER_KEY, MlflowModelAdapter


class ModelPublisher:
    """Publishes the best config as a new version in the MLflow model registry.

    The model is retrained on every row up to the latest as-of date, logged as
    an MLflow pyfunc model with the package code bundled in, registered under
    one name, and given an alias such as "champion" so other jobs can load the
    current best model with models:/<name>@champion.
    """

    def __init__(self, provider: ForecastDataProvider, registered_model_name: str, alias: str):
        self.provider = provider
        self.registered_model_name = registered_model_name
        self.alias = alias

    def publish(self, best: BacktestReport, as_of: pd.Timestamp | None = None) -> str:
        """Train, log, and register the best model. Returns the new version number."""
        as_of = pd.Timestamp(as_of) if as_of is not None else self.provider.last_date
        window = self.provider.window(as_of)
        model = ModelCatalog.get(best.model_name)(best.hyperparameters)
        model.train(window.training_data)

        example_input = window.future_features.reset_index(names="date")
        example_output = (
            model.predict(window.future_features).rename_axis("date").reset_index(name="prediction")
        )

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
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"pandas=={pd.__version__}",
                    f"numpy=={np.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"joblib=={joblib.__version__}",
                ],
                registered_model_name=self.registered_model_name,
            )

        mlflow.set_tags({"model_name": best.model_name, "stage": "publish"})
        mlflow.log_params({"model_name": best.model_name, **best.hyperparameters})
        mlflow.log_param("trained_through", window.training_data.index.max().date())
        mlflow.log_metrics({f"backtest_{name}": value for name, value in best.metrics().items()})

        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(self.registered_model_name, self.alias, version)
        client.update_model_version(
            self.registered_model_name,
            version,
            description=(
                f"{best.model_name} {json.dumps(best.hyperparameters, sort_keys=True)}, "
                f"trained through {window.training_data.index.max().date()}"
            ),
        )
        for name, value in best.metrics().items():
            client.set_model_version_tag(
                self.registered_model_name, version, f"backtest_{name}", f"{value:.4f}"
            )
        return version

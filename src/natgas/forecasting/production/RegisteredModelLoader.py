"""Loads a model version from the MLflow registry by alias."""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper


class RegisteredModelLoader:
    """Returns the version number and model config behind an alias like champion."""

    def __init__(self, tracking_uri: str, registered_model_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.registered_model_name = registered_model_name

    def load(self, alias: str) -> tuple[str, AbstractConfigWrapper]:
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(self.tracking_uri)
        version = str(client.get_model_version_by_alias(self.registered_model_name, alias).version)
        pyfunc = mlflow.pyfunc.load_model(f"models:/{self.registered_model_name}/{version}")
        return version, pyfunc.unwrap_python_model().model.config

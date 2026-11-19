"""Reads the latest grid search and its best configs from MLflow."""

from __future__ import annotations

from dataclasses import fields
from typing import get_type_hints

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper
from natgas.forecasting.models.ModelCatalog import ModelCatalog

GRID_EXPERIMENT = "natgas-grid-search"
FORECASTS_FILE = "backtest_forecasts.csv"


class GridSearchRuns:
    """The grid search results that the backtest export and the challenger use.

    A grid search logs one parent run, one nested run per config with its
    scores and params, and each config's backtest forecasts as a CSV artifact.
    This class finds the newest finished search for a horizon, the best nested
    run per model, the config behind a run, and the registry version that was
    published from the search, if any.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = GRID_EXPERIMENT,
        horizon_days: int = 30,
        selection_metric: str = "mae",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri)
        self.experiment_name = experiment_name
        self.horizon_days = horizon_days
        self.selection_metric = selection_metric

    def latest_search(self) -> Run:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"No MLflow experiment named {self.experiment_name}")
        runs = self.client.search_runs(
            [experiment.experiment_id],
            "tags.method = 'grid'",
            order_by=["attributes.start_time DESC"],
        )
        for run in runs:
            # Only the search run itself gets configs_tried, once every config is scored
            finished = "configs_tried" in run.data.tags
            if finished and run.data.params.get("horizon_days") == str(self.horizon_days):
                return run
        raise ValueError(
            f"No finished {self.horizon_days} day grid search in {self.experiment_name}"
        )

    def best_runs(self, search: Run) -> dict[str, Run]:
        """The lowest scoring nested run for each model. Failed configs have no score."""
        children = self.client.search_runs(
            [search.info.experiment_id],
            f"tags.mlflow.parentRunId = '{search.info.run_id}'",
            max_results=5000,
        )
        best: dict[str, Run] = {}
        for run in children:
            score = run.data.metrics.get(self.selection_metric)
            if score is None:
                continue
            name = run.data.tags["model_name"]
            if name not in best or score < best[name].data.metrics[self.selection_metric]:
                best[name] = run
        return best

    def forecasts(self, run: Run) -> pd.DataFrame:
        """The backtest forecasts logged with a run: date, as_of, step, actual, prediction."""
        path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path=FORECASTS_FILE, tracking_uri=self.tracking_uri
        )
        return pd.read_csv(path, parse_dates=["date", "as_of"])

    @staticmethod
    def config_for(run: Run) -> AbstractConfigWrapper:
        """Rebuild a config from the params MLflow stored as text."""
        config_class = ModelCatalog.config_class(run.data.tags["model_name"])
        types = get_type_hints(config_class)
        params = run.data.params
        values = {
            field.name: GridSearchRuns.parse(params[field.name], types[field.name])
            for field in fields(config_class)
            if field.name in params
        }
        return config_class.from_dict(values)

    @staticmethod
    def parse(text: str, kind: type):
        if kind is bool:
            return text == "True"
        if kind in (int, float):
            return kind(text)
        if kind == list[str]:
            return text.split(",") if text else []
        return text

    def registered_versions(self, search: Run, registered_model_name: str) -> dict[str, str]:
        """Model name to the newest registry version published from this search."""
        versions: dict[str, str] = {}
        for version in self.client.search_model_versions(f"name='{registered_model_name}'"):
            tags = self.client.get_run(version.run_id).data.tags
            if tags.get("search_run_id") != search.info.run_id:
                continue
            name = tags["model_name"]
            if name not in versions or int(version.version) > int(versions[name]):
                versions[name] = str(version.version)
        return versions

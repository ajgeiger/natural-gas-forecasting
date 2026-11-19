"""Registers the best config of one model from the grid search as the challenger."""

from __future__ import annotations

import mlflow
import pandas as pd

from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.production.GridSearchRuns import GridSearchRuns
from natgas.forecasting.tuning.ModelPublisher import BASELINE_MODEL, ModelPublisher
from natgas.forecasting.tuning.PublishDecision import PublishDecision

CHALLENGER_ALIAS = "challenger"
# A rerun must land this close to the logged score to be trusted
MATCH_TOLERANCE = 1e-6


class ChallengerPublisher:
    """Publishes a second model next to the champion, under the challenger alias.

    The challenger is scored every day as the staging model, so the dashboard
    can show it beside production. The config comes from the latest grid
    search. It is backtested again on the same as-of dates to get a report for
    the publisher, and publishing stops if the score does not match the one in
    MLflow. The champion alias is not touched.
    """

    def __init__(
        self,
        runs: GridSearchRuns,
        provider: ForecastDataProvider,
        registered_model_name: str,
        model_name: str = "gradient_boosting",
        alias: str = CHALLENGER_ALIAS,
        n_jobs: int = 1,
    ) -> None:
        self.runs = runs
        self.provider = provider
        self.registered_model_name = registered_model_name
        self.model_name = model_name
        self.alias = alias
        self.n_jobs = n_jobs

    def publish(self) -> str:
        """Returns the new registry version."""
        metric = self.runs.selection_metric
        search = self.runs.latest_search()
        best = self.runs.best_runs(search)
        run = best[self.model_name]
        config = self.runs.config_for(run)

        logged_dates = self.runs.forecasts(run)["as_of"].unique()
        as_of_dates = sorted(pd.Timestamp(day) for day in logged_dates)
        report = Backtester(self.provider, as_of_dates, self.n_jobs).run(config)
        logged = run.data.metrics[metric]
        rerun = report.metrics()[metric]
        if abs(rerun - logged) > MATCH_TOLERANCE:
            raise ValueError(f"{self.model_name} rerun scored {rerun}, the grid search {logged}")

        improvement = None
        if BASELINE_MODEL in best:
            improvement = 1 - logged / best[BASELINE_MODEL].data.metrics[metric]
        reason = f"best {self.model_name} config in the grid search, set as {self.alias}"

        mlflow.set_tracking_uri(self.runs.tracking_uri)
        mlflow.set_experiment(self.runs.experiment_name)
        publisher = ModelPublisher(self.provider, self.registered_model_name, self.alias, metric)
        with mlflow.start_run(run_name=f"publish-{self.alias}-{self.model_name}"):
            mlflow.set_tags({"search_run_id": search.info.run_id, "method": "grid"})
            return publisher.publish(PublishDecision(report, reason, improvement))

"""Rescores the backtest tags on every tournament version up to a cutoff."""

from __future__ import annotations

import logging
from datetime import date

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from backtesting.BacktestReport import HORIZON_BUCKETS
from backtesting.ForecastMetrics import ForecastMetrics
from production.registry.GridSearchRuns import FORECASTS_FILE
from production.tournament.TournamentConfig import TournamentConfig

log = logging.getLogger(__name__)

TAG_PREFIX = "backtest_"
CUTOFF_TAG = "backtest_cutoff"
ROWS_TAG = "backtest_rows"


class BacktestTagRefresher:
    """Rewrites the backtest_ tags of each registered version from its logged forecasts.

    The scorecard falls back on these tags until a model has 30 production
    days. When past days are replayed as production, a backtest that saw the
    prices of those days would hand the model knowledge it could not have had
    back then. So only forecasts whose target date is before the cutoff
    count. Their actual price was known before the first replayed day, which
    also rules out every backtest made on or after the cutoff.

    The forecasts come from the backtest_forecasts.csv artifact of the grid
    search run the version was seeded from (its source_run_id tag), or from
    the version's own run. A version with neither loses its backtest_ tags, so
    the scorecard uses the naive backtest instead of a stale score.
    """

    def __init__(self, config: TournamentConfig) -> None:
        self.config = config
        self.client = MlflowClient(config.tracking_uri)

    @property
    def registered_models(self) -> list[str]:
        names = [self.config.registered_model(family) for family in self.config.families]
        return [*names, self.config.registered_model(self.config.baseline)]

    @staticmethod
    def metrics(forecasts: pd.DataFrame) -> dict[str, float]:
        """Overall and per bucket scores, named like the published tags."""
        scores = dict(ForecastMetrics.score(forecasts["actual"], forecasts["prediction"]))
        for first, last in HORIZON_BUCKETS:
            group = forecasts[forecasts["step"].between(first, last)]
            if len(group):
                bucket = ForecastMetrics.score(group["actual"], group["prediction"])
                for name, value in bucket.items():
                    scores[f"{name}_days_{first}_to_{last}"] = value
        return scores

    @staticmethod
    def before(forecasts: pd.DataFrame, cutoff: date) -> pd.DataFrame:
        """Forecasts whose target date, and so also as-of date, is before the cutoff."""
        cutoff = pd.Timestamp(cutoff)
        known = (pd.to_datetime(forecasts["date"]) < cutoff) & (
            pd.to_datetime(forecasts["as_of"]) < cutoff
        )
        return forecasts[known]

    def forecasts_for(self, run_id: str) -> pd.DataFrame | None:
        """The logged backtest forecasts of the version's grid search run, if any."""
        run = self.client.get_run(run_id)
        candidates = [run.data.tags.get("source_run_id"), run_id]
        for candidate in [item for item in candidates if item]:
            try:
                path = mlflow.artifacts.download_artifacts(
                    run_id=candidate,
                    artifact_path=FORECASTS_FILE,
                    tracking_uri=self.config.tracking_uri,
                )
            except (MlflowException, OSError):
                continue
            return pd.read_csv(path, parse_dates=["date", "as_of"])
        return None

    def clear_tags(self, name: str, version: str) -> None:
        tags = self.client.get_model_version(name, version).tags
        for key in tags:
            if key.startswith(TAG_PREFIX):
                self.client.delete_model_version_tag(name, version, key)

    def refresh_version(self, name: str, version: str, run_id: str, cutoff: date) -> dict:
        """New tags for one version. Returns what was written."""
        forecasts = self.forecasts_for(run_id)
        known = self.before(forecasts, cutoff) if forecasts is not None else None
        self.clear_tags(name, version)
        if known is None or known.empty:
            log.warning("%s v%s has no backtest before %s, tags removed", name, version, cutoff)
            return {"name": name, "version": version, "rows": 0, "metrics": {}}
        metrics = self.metrics(known)
        for metric, value in metrics.items():
            self.client.set_model_version_tag(
                name, version, f"{TAG_PREFIX}{metric}", f"{value:.6f}"
            )
        self.client.set_model_version_tag(
            name, version, CUTOFF_TAG, str(pd.Timestamp(cutoff).date())
        )
        self.client.set_model_version_tag(name, version, ROWS_TAG, str(len(known)))
        log.info("%s v%s: mae %.6f from %d rows", name, version, metrics["mae"], len(known))
        return {"name": name, "version": version, "rows": len(known), "metrics": metrics}

    def refresh(self, cutoff: date) -> list[dict]:
        """Rescore every version of every tournament registry."""
        refreshed = []
        for name in self.registered_models:
            versions = self.client.search_model_versions(f"name='{name}'")
            for item in sorted(versions, key=lambda version: int(version.version)):
                refreshed.append(self.refresh_version(name, str(item.version), item.run_id, cutoff))
        baseline = self.config.registered_model(self.config.baseline)
        if not any(item["rows"] for item in refreshed if item["name"] == baseline):
            raise ValueError(f"{baseline} has no backtest before {cutoff}")
        return refreshed

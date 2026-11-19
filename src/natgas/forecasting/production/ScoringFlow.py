"""Prefect orchestration of the daily scoring job, for run history and logs in the Prefect UI."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.logging.handlers import APILogHandler
from prefect.schedules import Cron

from natgas.config import load_settings
from natgas.forecasting.production.ScoringPipeline import ScoringPipeline
from natgas.forecasting.production.ScoringRun import ScoringRun
from natgas.forecasting.tuning.TuningConfig import TuningConfig

FLOW_NAME = "natgas-daily-scoring"
DEPLOYMENT_NAME = "natgas-daily-scoring"
DEFAULT_CONFIG = str(Path("config") / "tuning.yaml")
DEFAULT_CRON = "0 7 * * 2-6"
DEFAULT_TIMEZONE = "America/New_York"


class ScoringFlow:
    """Runs ScoringPipeline as a Prefect flow with one task per stage.

    The tasks only call the pipeline's stage methods, so the business logic
    stays in ScoringPipeline and `natgas score` and `natgas flow` write the
    same rows. Reruns stay safe for the same reason: a stage already scored
    for the as-of date is skipped by the prediction store.

    refresh_data retries twice, 30 seconds apart, because Yahoo, Open-Meteo,
    and EIA sometimes fail for a moment. Scoring and actuals do not retry: a
    failure there is a real problem and should show as failed in the UI.

    Tasks use NO_CACHE, so every run does the work again instead of reusing
    an earlier result, and the pipeline object is never hashed.
    """

    @staticmethod
    def pipeline_for(config: str, refresh: bool) -> ScoringPipeline:
        return ScoringPipeline(load_settings(), TuningConfig.from_yaml(config), refresh=refresh)

    @staticmethod
    def forward_logs() -> None:
        """Send natgas log records to the Prefect API, so they show on the run page."""
        target = logging.getLogger("natgas")
        handlers = logging.getLogger("prefect.extra").handlers
        for handler in handlers:
            if isinstance(handler, APILogHandler) and handler not in target.handlers:
                target.addHandler(handler)

    @staticmethod
    def summary(run: ScoringRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "run_timestamp": run.run_timestamp.isoformat(),
            "as_of_date": run.as_of_date.date().isoformat(),
            "loaded": dict(run.loaded),
            "saved": dict(run.saved),
            "weather_days": dict(run.weather_days),
            "actuals": run.actuals,
        }

    @staticmethod
    @task(name="refresh_data", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
    def refresh_data(pipeline: ScoringPipeline) -> dict[str, int]:
        if not pipeline.refresh:
            print("refresh skipped, scoring from the data already loaded")
        loaded = pipeline.refresh_data()
        print(f"loaded: {loaded}")
        return loaded

    @staticmethod
    @task(name="score_models", cache_policy=NO_CACHE)
    def score_models(pipeline: ScoringPipeline) -> ScoringRun:
        scored = pipeline.score_models()
        print(f"as of {scored.as_of_date.date()}: predictions saved {scored.saved}")
        print(f"future weather days: {scored.weather_days}")
        return scored

    @staticmethod
    @task(name="backfill_actuals", cache_policy=NO_CACHE)
    def backfill_actuals(pipeline: ScoringPipeline, as_of_date: date) -> int:
        actuals = pipeline.backfill_actuals(as_of_date)
        print(f"matured forecasts in forecast_actuals: {actuals}")
        return actuals

    @staticmethod
    @flow(name=FLOW_NAME, log_prints=True)
    def daily_scoring(refresh: bool = True, config: str = DEFAULT_CONFIG) -> dict[str, Any]:
        """Refresh data, score production and staging, and backfill actuals."""
        ScoringFlow.forward_logs()
        pipeline = ScoringFlow.pipeline_for(config, refresh)
        loaded = ScoringFlow.refresh_data(pipeline)
        scored = ScoringFlow.score_models(pipeline)
        actuals = ScoringFlow.backfill_actuals(pipeline, scored.as_of_date.date())
        run = ScoringRun(
            run_id=scored.run_id,
            run_timestamp=scored.run_timestamp,
            as_of_date=scored.as_of_date,
            loaded=loaded,
            saved=scored.saved,
            actuals=actuals,
            weather_days=scored.weather_days,
        )
        summary = ScoringFlow.summary(run)
        print(
            f"run {summary['run_id']} as of {summary['as_of_date']}: "
            f"saved {summary['saved']}, weather {summary['weather_days']}, "
            f"{summary['actuals']} matured"
        )
        return summary

    def __init__(self, config: str = DEFAULT_CONFIG) -> None:
        self.config = config

    def run_once(self, refresh: bool = True) -> dict[str, Any]:
        return ScoringFlow.daily_scoring(refresh=refresh, config=self.config)

    def serve(self, cron: str = DEFAULT_CRON, timezone: str = DEFAULT_TIMEZONE) -> None:
        """Serve the scheduled deployment until stopped. Needs a running Prefect server.

        Prefect 3 Flow.serve has no timezone argument, so the cron and its
        timezone go in a Cron schedule. global_limit=1 keeps two runs from
        overlapping, the same guard the cron script gets from flock.
        """
        ScoringFlow.daily_scoring.serve(
            name=DEPLOYMENT_NAME,
            schedule=Cron(cron, timezone=timezone),
            parameters={"refresh": True, "config": self.config},
            description="Refresh data, score production and staging, backfill actuals.",
            tags=["natgas", "production"],
            global_limit=1,
        )

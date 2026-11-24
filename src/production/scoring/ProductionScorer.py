"""Scores every tournament contender for one as-of date."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from backtesting.Backtester import Backtester
from data.provider.ForecastDataProvider import ForecastDataProvider
from models.config.AbstractConfigWrapper import AbstractConfigWrapper
from models.ModelCatalog import ModelCatalog
from production.scoring.PredictionStore import PredictionStore
from production.tournament.Contender import Contender
from production.tournament.ModelRules import ModelRules

log = logging.getLogger(__name__)


class ProductionScorer:
    """Forecasts from one as-of date with each contender and saves them under its stage.

    Contenders are the family champions and challengers and the naive
    baseline, each loaded from its registry alias. The config of that
    version is refit on data through the as-of date, the same way every
    backtest did. Chronos-2 is zero-shot, so the refit only swaps in the
    latest prices. A stage that already has forecasts for the as-of date is
    skipped, so reruns add nothing.

    Two rules are checked before scoring: Chronos-2 must use weather, and
    Greykite must not. A contender that breaks a rule is not scored, and it
    cannot qualify for an event.
    """

    def __init__(self, provider: ForecastDataProvider, store: PredictionStore) -> None:
        self.provider = provider
        self.store = store

    @staticmethod
    def check_rules(config: AbstractConfigWrapper) -> None:
        ModelRules.check(config)

    def forecast(self, config: AbstractConfigWrapper, as_of: pd.Timestamp) -> pd.DataFrame:
        """prediction, lower, and upper for each forecast date."""
        Backtester(self.provider, [as_of]).check_config(config)
        window = self.provider.window(as_of)
        Backtester.check_no_leakage(window.as_of, window.training_data, window.forecast_dates)
        model = ModelCatalog.build(config)
        model.train(window.training_data)
        return model.predict_with_interval(window.future_features)

    @staticmethod
    def rows_for(
        contender: Contender,
        forecast: pd.DataFrame,
        as_of: pd.Timestamp,
        run_id: str,
        run_timestamp: datetime,
    ) -> list[dict]:
        return [
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "as_of_date": as_of.date(),
                "target_date": day.date(),
                "horizon_day": step,
                "model": contender.model,
                "model_version": str(contender.version),
                "stage": contender.stage,
                "forecast": float(values["prediction"]),
                "lower": None if pd.isna(values["lower"]) else float(values["lower"]),
                "upper": None if pd.isna(values["upper"]) else float(values["upper"]),
            }
            for step, (day, values) in enumerate(forecast.iterrows(), start=1)
        ]

    def score(
        self,
        as_of: pd.Timestamp,
        run_id: str,
        run_timestamp: datetime,
        contenders: list[Contender],
    ) -> dict[str, int]:
        """Rows saved per stage. Zero means the stage was already scored or broke a rule."""
        as_of = pd.Timestamp(as_of).normalize()
        saved = {}
        for contender in contenders:
            stage = contender.stage
            if self.store.has_run(as_of.date(), stage):
                log.info("%s already scored for %s, skipping", stage, as_of.date())
                saved[stage] = 0
                continue
            problem = ModelRules.problem(contender.config)
            if problem:
                log.error("%s not scored: %s", contender.label, problem)
                saved[stage] = 0
                continue
            forecast = self.forecast(contender.config, as_of)
            rows = self.rows_for(contender, forecast, as_of, run_id, run_timestamp)
            saved[stage] = self.store.append(rows)
            log.info("%s: %d rows", contender.label, len(rows))
        return saved

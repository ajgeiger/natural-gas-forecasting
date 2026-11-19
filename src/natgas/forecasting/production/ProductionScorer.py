"""Scores the registered production and staging models for one as-of date."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from natgas.forecasting.backtesting.Backtester import Backtester
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.data.WeatherFeatureSource import COLUMNS as WEATHER_COLUMNS
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.production.PredictionStore import PredictionStore
from natgas.forecasting.production.RegisteredModelLoader import RegisteredModelLoader

log = logging.getLogger(__name__)

# Stage name in the predictions table and the registry alias it scores
STAGE_ALIASES = {"production": "champion", "staging": "challenger"}


class ProductionScorer:
    """Forecasts from one as-of date with each registered model and saves them.

    Each stage loads its registry alias (production is champion, staging is
    challenger), takes the config of that version, and refits it on data
    through the as-of date, the same way every backtest did. Chronos-2 is
    zero-shot, so the refit only swaps in the latest prices. A stage that
    already has forecasts for the as-of date is skipped, so reruns add nothing.

    Two rules are checked before scoring: Chronos-2 must use weather, and
    Greykite must not.
    """

    def __init__(
        self,
        provider: ForecastDataProvider,
        loader: RegisteredModelLoader,
        store: PredictionStore,
        stage_aliases: dict[str, str] = STAGE_ALIASES,
    ) -> None:
        self.provider = provider
        self.loader = loader
        self.store = store
        self.stage_aliases = stage_aliases

    @staticmethod
    def check_rules(config: AbstractConfigWrapper) -> None:
        uses_weather = bool(set(config.required_features) & set(WEATHER_COLUMNS))
        if config.model_name == "chronos" and not uses_weather:
            raise ValueError("Chronos-2 must use weather covariates")
        if config.model_name == "greykite" and uses_weather:
            raise ValueError("Greykite must not use weather")

    def forecast(self, config: AbstractConfigWrapper, as_of: pd.Timestamp) -> pd.DataFrame:
        """prediction, lower, and upper for each forecast date."""
        Backtester(self.provider, [as_of]).check_config(config)
        window = self.provider.window(as_of)
        Backtester.check_no_leakage(window.as_of, window.training_data, window.forecast_dates)
        model = ModelCatalog.build(config)
        model.train(window.training_data)
        return model.predict_with_interval(window.future_features)

    def score(self, as_of: pd.Timestamp, run_id: str, run_timestamp: datetime) -> dict[str, int]:
        """Rows saved per stage. Zero means the stage already had this as-of date."""
        as_of = pd.Timestamp(as_of).normalize()
        saved = {}
        for stage, alias in self.stage_aliases.items():
            if self.store.has_run(as_of.date(), stage):
                log.info("%s already scored for %s, skipping", stage, as_of.date())
                saved[stage] = 0
                continue
            version, config = self.loader.load(alias)
            self.check_rules(config)
            forecast = self.forecast(config, as_of)
            rows = [
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "as_of_date": as_of.date(),
                    "target_date": day.date(),
                    "horizon_day": step,
                    "model": config.model_name,
                    "model_version": version,
                    "stage": stage,
                    "forecast": float(values["prediction"]),
                    "lower": None if pd.isna(values["lower"]) else float(values["lower"]),
                    "upper": None if pd.isna(values["upper"]) else float(values["upper"]),
                }
                for step, (day, values) in enumerate(forecast.iterrows(), start=1)
            ]
            saved[stage] = self.store.append(rows)
            log.info("%s: %s version %s, %d rows", stage, config.model_name, version, len(rows))
        return saved

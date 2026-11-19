"""The daily production job: refresh data, score, and backfill actuals."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from natgas.config import Settings
from natgas.db import init_db
from natgas.db.models import utc_now
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.data.ForecastWeatherFeatureSource import ForecastWeatherFeatureSource
from natgas.forecasting.production.ActualsRefresher import ActualsRefresher
from natgas.forecasting.production.DataRefresher import DataRefresher
from natgas.forecasting.production.ForecastMonitor import ForecastMonitor
from natgas.forecasting.production.PredictionStore import PredictionStore
from natgas.forecasting.production.ProductionScorer import ProductionScorer
from natgas.forecasting.production.RegisteredModelLoader import RegisteredModelLoader
from natgas.forecasting.production.ScoringRun import ScoringRun
from natgas.forecasting.tuning.TuningConfig import TuningConfig

log = logging.getLogger(__name__)

MARKET_TIMEZONE = ZoneInfo("America/New_York")


class ScoringPipeline:
    """Runs one production scoring pass.

    1. Refresh prices, weather (with the 14 day forecast), and storage
    2. Pick the as-of date: the latest trading day before today. Today's bar
       is still trading, so it is never used, even after the close.
    3. Score production (champion) and staging (challenger) into predictions
    4. Copy matured forecasts with their actual prices into forecast_actuals
    5. Recreate the accuracy views

    Safe to run more than once a day: a stage already scored for the as-of
    date is skipped, and the actuals step only rewrites matured rows.
    """

    def __init__(
        self,
        settings: Settings,
        tuning: TuningConfig,
        today: date | None = None,
        refresh: bool = True,
    ) -> None:
        self.settings = settings
        self.tuning = tuning
        self.today = today or datetime.now(MARKET_TIMEZONE).date()
        self.refresh = refresh

    @staticmethod
    def last_complete_day(provider: ForecastDataProvider, today: date) -> pd.Timestamp:
        days = provider.target.index[provider.target.index < pd.Timestamp(today)]
        if len(days) == 0:
            raise ValueError(f"No prices before {today}")
        return days[-1]

    @staticmethod
    def weather_days(provider: ForecastDataProvider, as_of: pd.Timestamp) -> dict[str, int]:
        """How many forecast dates got forecast weather and how many got normals."""
        for source in provider.feature_sources:
            if isinstance(source, ForecastWeatherFeatureSource):
                kinds = source.kinds_for(provider.forecast_dates(as_of), as_of)
                return kinds.value_counts().to_dict()
        return {}

    def refresh_data(self) -> dict[str, int]:
        """Step 1: rows loaded per source, or nothing when refresh is off."""
        init_db(self.settings.db_url)
        if not self.refresh:
            return {}
        return DataRefresher(self.settings, self.today).refresh()

    def score_models(self) -> ScoringRun:
        """Steps 2 and 3: score both stages. Loaded and actuals are left empty."""
        db_url = self.settings.db_url
        init_db(db_url)
        provider = ForecastDataProvider.from_settings(
            self.settings, self.tuning.horizon_days, self.tuning.target_column, live_weather=True
        )
        as_of = self.last_complete_day(provider, self.today)
        run_id = uuid4().hex
        run_timestamp = utc_now()
        loader = RegisteredModelLoader(self.tuning.tracking_uri, self.tuning.registered_model_name)
        scorer = ProductionScorer(provider, loader, PredictionStore(db_url))
        saved = scorer.score(as_of, run_id, run_timestamp)
        return ScoringRun(
            run_id=run_id,
            run_timestamp=run_timestamp,
            as_of_date=as_of,
            loaded={},
            saved=saved,
            actuals=0,
            weather_days=self.weather_days(provider, as_of),
        )

    def backfill_actuals(self, as_of_date: date) -> int:
        """Steps 4 and 5: matured forecasts copied, then the accuracy views."""
        db_url = self.settings.db_url
        actuals = ActualsRefresher(db_url, self.tuning.target_column).refresh(as_of_date)
        ForecastMonitor(db_url).create_views()
        return actuals

    def run(self) -> ScoringRun:
        loaded = self.refresh_data()
        scored = self.score_models()
        actuals = self.backfill_actuals(scored.as_of_date.date())
        run = replace(scored, loaded=loaded, actuals=actuals)
        log.info(
            "run %s as of %s: saved %s, %d matured",
            run.run_id,
            run.as_of_date.date(),
            run.saved,
            run.actuals,
        )
        return run

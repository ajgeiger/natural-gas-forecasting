"""The daily production job: refresh data, score the contenders, backfill actuals, and
run the tournament."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from config import Settings
from data.database import init_db
from data.database.models import utc_now
from data.features.ForecastWeatherFeatureSource import ForecastWeatherFeatureSource
from data.provider.ForecastDataProvider import ForecastDataProvider
from optimization.config.TuningConfig import TuningConfig
from production.migrations.StageColumnMigrator import StageColumnMigrator
from production.publishing.ActualsRefresher import ActualsRefresher
from production.publishing.ForecastMonitor import ForecastMonitor
from production.scoring.DataRefresher import DataRefresher
from production.scoring.PredictionStore import PredictionStore
from production.scoring.ProductionScorer import ProductionScorer
from production.scoring.ScoringRun import ScoringRun
from production.tournament.FamilyRegistry import FamilyRegistry
from production.tournament.ModelTournament import ModelTournament
from production.tournament.TournamentConfig import TournamentConfig
from production.tournament.TournamentOutcome import TournamentOutcome

log = logging.getLogger(__name__)

MARKET_TIMEZONE = ZoneInfo("America/New_York")


class ScoringPipeline:
    """Runs one production scoring pass.

    1. Refresh prices, weather (with the 14 day forecast), and storage
    2. Pick the as-of date: the latest trading day before today. Today's bar
       is still trading, so it is never used, even after the close. Score
       every family champion and challenger and the naive baseline into
       predictions, each under its own stage
    3. Copy matured forecasts with their actual prices into forecast_actuals
       and recreate the accuracy views
    4. Run the tournament: judge the 60 day trials and promote, score every
       contender on the blended scorecard, pick the event winners, and
       publish their stitched forecast as stage production

    Safe to run more than once a day: a stage already scored for the as-of
    date is skipped, the actuals step only rewrites matured rows, and the
    tournament tables are keyed by as-of date.
    """

    def __init__(
        self,
        settings: Settings,
        tuning: TuningConfig,
        today: date | None = None,
        refresh: bool = True,
        tournament: TournamentConfig | None = None,
    ) -> None:
        self.settings = settings
        self.tuning = tuning
        self.today = today or datetime.now(MARKET_TIMEZONE).date()
        self.refresh = refresh
        self.tournament = tournament or TournamentConfig.from_tuning(tuning)

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

    def prepare_database(self) -> None:
        """Create missing tables and widen old stage columns."""
        init_db(self.settings.db_url)
        StageColumnMigrator(self.settings.db_url).migrate()

    def refresh_data(self) -> dict[str, int]:
        """Step 1: rows loaded per source, or nothing when refresh is off."""
        self.prepare_database()
        if not self.refresh:
            return {}
        return DataRefresher(self.settings, self.today).refresh()

    def score_models(self) -> ScoringRun:
        """Step 2: score every contender. Loaded and actuals are left empty."""
        db_url = self.settings.db_url
        self.prepare_database()
        provider = ForecastDataProvider.from_settings(
            self.settings, self.tuning.horizon_days, self.tuning.target_column, live_weather=True
        )
        as_of = self.last_complete_day(provider, self.today)
        run_id = uuid4().hex
        run_timestamp = utc_now()
        contenders = FamilyRegistry(self.tournament).load_all()
        scorer = ProductionScorer(provider, PredictionStore(db_url))
        saved = scorer.score(as_of, run_id, run_timestamp, contenders)
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
        """Step 3: matured forecasts copied, then the accuracy views."""
        db_url = self.settings.db_url
        actuals = ActualsRefresher(db_url, self.tuning.target_column).refresh(as_of_date)
        ForecastMonitor(db_url).create_views()
        return actuals

    def run_tournament(self, scored: ScoringRun) -> TournamentOutcome:
        """Step 4: trials, scorecard, events, and the stitched production forecast."""
        tournament = ModelTournament(self.tournament, self.settings.db_url)
        return tournament.run(scored.as_of_date.date(), scored.run_id, scored.run_timestamp)

    def run(self) -> ScoringRun:
        loaded = self.refresh_data()
        scored = self.score_models()
        actuals = self.backfill_actuals(scored.as_of_date.date())
        outcome = self.run_tournament(scored)
        run = replace(
            scored,
            loaded=loaded,
            actuals=actuals,
            events=outcome.winners(),
            promoted=outcome.promoted,
            stitched=outcome.stitched,
        )
        log.info(
            "run %s as of %s: saved %s, %d matured, winners %s, promoted %s",
            run.run_id,
            run.as_of_date.date(),
            run.saved,
            run.actuals,
            run.events,
            run.promoted,
        )
        return run

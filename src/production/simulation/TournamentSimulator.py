"""Replays past trading days through the production job, one day at a time."""

from __future__ import annotations

import logging
from datetime import date
from uuid import uuid4

import pandas as pd
from sqlalchemy import text

from config import Settings
from data.database import get_engine, init_db
from data.database.models import utc_now
from data.database.SqlReader import SqlReader
from data.features.AbstractFeatureSource import AbstractFeatureSource
from data.features.CalendarFeatureSource import CalendarFeatureSource
from data.features.ForecastWeatherFeatureSource import ForecastWeatherFeatureSource
from data.features.HolidayFeatureSource import HolidayFeatureSource
from data.features.StorageFeatureSource import (
    RELEASE_LAG_DAYS,
    StorageFeatureSource,
)
from data.features.WeatherFeatureSource import WeatherFeatureSource
from data.provider.ForecastDataProvider import PRICE_COLUMNS, ForecastDataProvider
from production.migrations.StageColumnMigrator import StageColumnMigrator
from production.publishing.ActualsRefresher import ActualsRefresher
from production.publishing.ForecastMonitor import ForecastMonitor
from production.scoring.PredictionStore import PredictionStore
from production.scoring.ProductionScorer import ProductionScorer
from production.scoring.ScoringRun import ScoringRun
from production.tournament.FamilyRegistry import FamilyRegistry
from production.tournament.ModelTournament import ModelTournament
from production.tournament.TournamentConfig import TournamentConfig

log = logging.getLogger(__name__)

SIM_PREFIX = "sim_"
RUN_ID_LENGTH = 32


class TournamentSimulator:
    """Builds a production history by replaying past days as if each one were today.

    For every simulated day, in order:

      1. Data is cut at the end of that day. Prices, observed weather, and
         storage after it are dropped before any model sees them. There is no
         stored weather forecast from back then, so every future date gets
         the 10 year smoothed daily normal, computed from weather observed on
         or before that day.
      2. Every contender is scored, the same way the daily job does it, under
         a run id that starts with sim_ so replayed runs are easy to tell apart
         from live ones.
      3. Forecasts whose target date is on or before that day get their actual
         price, and the accuracy views are rebuilt.
      4. The tournament runs for that day: trials, scorecard, events, and the
         stitched production forecast.

    Each day only sees what the days before it wrote, so the trials, the
    rolling scorecard, and any promotion play out as they would have live.
    """

    def __init__(
        self,
        settings: Settings,
        tournament: TournamentConfig,
        horizon_days: int = 30,
        target_column: str = "high",
        registry: FamilyRegistry | None = None,
    ) -> None:
        if target_column not in PRICE_COLUMNS:
            raise ValueError(f"target_column must be one of {PRICE_COLUMNS}")
        self.settings = settings
        self.tournament = tournament
        self.horizon_days = horizon_days
        self.target_column = target_column
        self.registry = registry or FamilyRegistry(tournament)
        engine = get_engine(settings.db_url)
        self.target = self.read_target(engine, target_column)
        self.weather = SqlReader.read(
            engine,
            text("SELECT date, location, hdd, cdd FROM weather_daily"),
            ["date", "location", "hdd", "cdd"],
        )
        self.weather["date"] = pd.to_datetime(self.weather["date"]).dt.normalize()
        self.storage = SqlReader.read(
            engine,
            text("SELECT week_ending, value_bcf FROM storage_weekly WHERE region = :region"),
            ["week_ending", "value_bcf"],
            {"region": settings.storage_region},
        )
        self.storage["week_ending"] = pd.to_datetime(self.storage["week_ending"])

    @staticmethod
    def read_target(engine, target_column: str) -> pd.Series:
        query = text(f"SELECT date, {target_column} AS value FROM prices ORDER BY date")
        prices = SqlReader.read(engine, query, ["date", "value"])
        prices["date"] = pd.to_datetime(prices["date"])
        return prices.set_index("date")["value"].dropna()

    def simulation_days(self, end: date, count: int) -> list[pd.Timestamp]:
        """The last count trading days with a price, ending on or before end."""
        days = self.target.index[self.target.index <= pd.Timestamp(end)]
        if len(days) < count:
            raise ValueError(f"only {len(days)} trading days on or before {end}")
        return list(days[-count:])

    @staticmethod
    def run_id_for(sim_day: pd.Timestamp) -> str:
        """sim_ plus the day plus random hex, 32 characters like a live run id."""
        prefix = f"{SIM_PREFIX}{pd.Timestamp(sim_day):%Y%m%d}_"
        return prefix + uuid4().hex[: RUN_ID_LENGTH - len(prefix)]

    def provider_for(self, sim_day: pd.Timestamp) -> ForecastDataProvider:
        """A data provider that holds nothing from after the end of sim_day."""
        sim_day = pd.Timestamp(sim_day).normalize()
        weather = self.weather[self.weather["date"] <= sim_day]
        sources: list[AbstractFeatureSource] = [
            ForecastWeatherFeatureSource(weather),
            HolidayFeatureSource(),
            CalendarFeatureSource(),
        ]
        storage = StorageFeatureSource(self.storage[self.storage["week_ending"] <= sim_day])
        if storage.has_data:
            sources.append(storage)
        target = self.target.loc[:sim_day]
        provider = ForecastDataProvider(target, sources, self.horizon_days)
        self.check_no_future_data(provider, sim_day)
        return provider

    @staticmethod
    def check_no_future_data(provider: ForecastDataProvider, sim_day: pd.Timestamp) -> None:
        """Fail loudly if any price, weather, or storage row is dated after sim_day.

        Holidays and the calendar are known ahead, so only these can leak.
        """
        day = sim_day.date()
        if provider.last_date > sim_day:
            raise ValueError(f"prices after {day} leaked into the replay")
        for source in provider.feature_sources:
            weather = isinstance(source, WeatherFeatureSource) and len(source.daily)
            if weather and source.daily.index.max() > sim_day:
                raise ValueError(f"weather after {day} leaked into the replay")
            storage = isinstance(source, StorageFeatureSource) and source.has_data
            lag = pd.Timedelta(days=RELEASE_LAG_DAYS)
            if storage and source.releases.index.max() - lag > sim_day:
                raise ValueError(f"storage after {day} leaked into the replay")

    @staticmethod
    def weather_days(provider: ForecastDataProvider, sim_day: pd.Timestamp) -> dict[str, int]:
        """How many forecast dates got observed, forecast, or normal weather."""
        for source in provider.feature_sources:
            if isinstance(source, ForecastWeatherFeatureSource):
                kinds = source.kinds_for(provider.forecast_dates(sim_day), sim_day)
                return kinds.value_counts().to_dict()
        return {}

    def prepare_database(self) -> None:
        init_db(self.settings.db_url)
        StageColumnMigrator(self.settings.db_url).migrate()

    def simulate_day(self, sim_day: pd.Timestamp) -> ScoringRun:
        """Score, backfill, and run the tournament for one replayed day."""
        db_url = self.settings.db_url
        sim_day = pd.Timestamp(sim_day).normalize()
        provider = self.provider_for(sim_day)
        run_id = self.run_id_for(sim_day)
        run_timestamp = utc_now()
        contenders = self.registry.load_all()
        scorer = ProductionScorer(provider, PredictionStore(db_url))
        saved = scorer.score(sim_day, run_id, run_timestamp, contenders)
        actuals = ActualsRefresher(db_url, self.target_column).refresh(sim_day.date())
        ForecastMonitor(db_url).create_views()
        outcome = ModelTournament(self.tournament, db_url, self.registry).run(
            sim_day.date(), run_id, run_timestamp
        )
        run = ScoringRun(
            run_id=run_id,
            run_timestamp=run_timestamp,
            as_of_date=sim_day,
            loaded={},
            saved=saved,
            actuals=actuals,
            weather_days=self.weather_days(provider, sim_day),
            events=outcome.winners(),
            promoted=outcome.promoted,
            stitched=outcome.stitched,
        )
        log.info(
            "simulated %s: %d matured, winners %s, promoted %s",
            sim_day.date(),
            actuals,
            run.events,
            run.promoted,
        )
        return run

    def run(self, end: date, count: int = 60) -> list[ScoringRun]:
        """Replay the last count trading days ending on end, oldest first."""
        self.prepare_database()
        return [self.simulate_day(day) for day in self.simulation_days(end, count)]

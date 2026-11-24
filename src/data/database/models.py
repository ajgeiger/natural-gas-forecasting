"""Database tables."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Long enough for stages like "gradient_boosting:challenger"
STAGE_LENGTH = 32


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Price(Base):
    """Daily futures prices, one row per trading day."""

    __tablename__ = "prices"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class WeatherDaily(Base):
    """Daily mean temperature and degree days per city."""

    __tablename__ = "weather_daily"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    location: Mapped[str] = mapped_column(String(64), primary_key=True)
    tavg_f: Mapped[float] = mapped_column(Float)
    hdd: Mapped[float] = mapped_column(Float)
    cdd: Mapped[float] = mapped_column(Float)
    is_forecast: Mapped[bool] = mapped_column(Boolean, default=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class StorageWeekly(Base):
    """EIA weekly working gas in storage."""

    __tablename__ = "storage_weekly"

    week_ending: Mapped[date] = mapped_column(Date, primary_key=True)
    region: Mapped[str] = mapped_column(String(32), primary_key=True)
    value_bcf: Mapped[float] = mapped_column(Float)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IngestionRun(Base):
    """One row per loader run, for monitoring."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="running")
    message: Mapped[str | None] = mapped_column(Text)


class BacktestForecast(Base):
    """Every backtest forecast for the best config of each model, for charts.

    Rewritten in full by BacktestExporter from the latest grid search.
    """

    __tablename__ = "backtest_forecasts"

    model: Mapped[str] = mapped_column(String(32), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    horizon_day: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date)
    model_version: Mapped[str] = mapped_column(String(64))
    forecast: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    actual: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    """Live forecasts, one run per as-of date and stage. Rows are only ever added.

    Stage "production" is the business facing forecast stitched from the
    tournament event winners. Every contender also gets its own stage, such
    as "chronos:champion", "greykite:challenger", or "baseline:naive". The key
    makes a second run for the same as-of date and stage fail instead of
    adding duplicates.
    """

    __tablename__ = "predictions"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    stage: Mapped[str] = mapped_column(String(STAGE_LENGTH), primary_key=True)
    horizon_day: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    run_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    target_date: Mapped[date] = mapped_column(Date)
    model: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    forecast: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)


class ForecastActual(Base):
    """Live forecasts whose target date has passed, with the actual price and error."""

    __tablename__ = "forecast_actuals"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    stage: Mapped[str] = mapped_column(String(STAGE_LENGTH), primary_key=True)
    horizon_day: Mapped[int] = mapped_column(Integer, primary_key=True)
    horizon_bucket: Mapped[str] = mapped_column(String(16))
    run_id: Mapped[str] = mapped_column(String(32))
    target_date: Mapped[date] = mapped_column(Date)
    model: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    forecast: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    actual: Mapped[float] = mapped_column(Float)
    error: Mapped[float] = mapped_column(Float)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TournamentStanding(Base):
    """Blended score of every contender in every horizon bucket, one row per as-of date.

    blended_mae = prod_weight * prod_mae + (1 - prod_weight) * backtest_mae,
    where prod_weight is the share of the 30 day window that has matured in
    production. qualified and is_winner are only set in the three events.
    """

    __tablename__ = "tournament_standings"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    horizon_bucket: Mapped[str] = mapped_column(String(16), primary_key=True)
    stage: Mapped[str] = mapped_column(String(STAGE_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32))
    family: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    registered_model: Mapped[str] = mapped_column(String(64))
    prod_days: Mapped[int] = mapped_column(Integer)
    prod_mae: Mapped[float] = mapped_column(Float)
    backtest_mae: Mapped[float] = mapped_column(Float)
    backtest_source: Mapped[str] = mapped_column(String(16))
    prod_weight: Mapped[float] = mapped_column(Float)
    blended_mae: Mapped[float] = mapped_column(Float)
    baseline_mae: Mapped[float] = mapped_column(Float)
    improvement: Mapped[float] = mapped_column(Float)
    rules_passed: Mapped[bool] = mapped_column(Boolean)
    qualified: Mapped[bool] = mapped_column(Boolean)
    rank: Mapped[int] = mapped_column(Integer)
    is_winner: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EventWinner(Base):
    """The winner of each horizon event, the model published for those days."""

    __tablename__ = "event_winners"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    horizon_bucket: Mapped[str] = mapped_column(String(16), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32))
    first_day: Mapped[int] = mapped_column(Integer)
    last_day: Mapped[int] = mapped_column(Integer)
    family: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(STAGE_LENGTH))
    model: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    registered_model: Mapped[str] = mapped_column(String(64))
    blended_mae: Mapped[float] = mapped_column(Float)
    baseline_mae: Mapped[float] = mapped_column(Float)
    improvement: Mapped[float] = mapped_column(Float)
    qualified_count: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ChallengerTrialRecord(Base):
    """Daily progress of each family's challenger through its 60 day proving ground."""

    __tablename__ = "challenger_trials"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    family: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32))
    registered_model: Mapped[str] = mapped_column(String(64))
    champion_version: Mapped[str] = mapped_column(String(64))
    challenger_version: Mapped[str] = mapped_column(String(64))
    trial_started: Mapped[date | None] = mapped_column(Date)
    trial_days: Mapped[int] = mapped_column(Integer)
    required_days: Mapped[int] = mapped_column(Integer)
    champion_mae: Mapped[float] = mapped_column(Float)
    challenger_mae: Mapped[float] = mapped_column(Float)
    challenger_ahead: Mapped[bool] = mapped_column(Boolean)
    promoted: Mapped[bool] = mapped_column(Boolean)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32))


class PromotionRecord(Base):
    """Every challenger that earned its family's champion title."""

    __tablename__ = "promotion_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    family: Mapped[str] = mapped_column(String(32))
    registered_model: Mapped[str] = mapped_column(String(64))
    new_champion_version: Mapped[str] = mapped_column(String(64))
    former_champion_version: Mapped[str] = mapped_column(String(64))
    trial_days: Mapped[int] = mapped_column(Integer)
    champion_mae: Mapped[float] = mapped_column(Float)
    challenger_mae: Mapped[float] = mapped_column(Float)
    run_id: Mapped[str] = mapped_column(String(32))
    promoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

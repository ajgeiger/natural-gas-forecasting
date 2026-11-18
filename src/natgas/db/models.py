"""Database tables."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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

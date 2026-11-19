"""Gives models their data for a forecast date, with no look-ahead."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from natgas.config import Settings
from natgas.db import get_engine
from natgas.forecasting.data.AbstractFeatureSource import AbstractFeatureSource
from natgas.forecasting.data.ForecastWindow import ForecastWindow
from natgas.forecasting.data.SqlReader import SqlReader
from natgas.forecasting.data.StorageFeatureSource import StorageFeatureSource
from natgas.forecasting.data.WeatherFeatureSource import WeatherFeatureSource

TARGET_COLUMN = "price"
PRICE_COLUMNS = ["open", "high", "low", "close"]


class ForecastDataProvider:
    """Single place where models get data.

    You pass an as-of date and get back a ForecastWindow with:
      - training data: the target and features on or before the as-of date
      - future features: feature values for the dates to predict, as known on the as-of date
      - test target: the actual values for those dates, used only to score the forecast

    Every backtest and every live forecast goes through this class, so the
    no look-ahead rule lives in one place.
    """

    def __init__(
        self,
        target: pd.Series,
        feature_sources: list[AbstractFeatureSource] | None = None,
        horizon_days: int = 5,
    ) -> None:
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        target = target.dropna().sort_index()
        target.index = pd.DatetimeIndex(target.index).normalize()
        self.target = target.rename(TARGET_COLUMN)
        self.feature_sources = feature_sources or []
        self.horizon_days = horizon_days

    @classmethod
    def from_settings(
        cls, settings: Settings, horizon_days: int = 5, target_column: str = "high"
    ) -> ForecastDataProvider:
        """Build a provider from the database: daily prices plus weather and storage."""
        if target_column not in PRICE_COLUMNS:
            raise ValueError(f"target_column must be one of {PRICE_COLUMNS}")
        engine = get_engine(settings.db_url)
        query = text(f"SELECT date, {target_column} AS value FROM prices ORDER BY date")
        prices = SqlReader.read(engine, query, ["date", "value"])
        prices["date"] = pd.to_datetime(prices["date"])
        target = prices.set_index("date")["value"]

        sources: list[AbstractFeatureSource] = [WeatherFeatureSource.from_database(engine)]
        storage = StorageFeatureSource.from_database(engine, settings.storage_region)
        # Storage needs an EIA key to load, so only use it when it has data
        if storage.has_data:
            sources.append(storage)
        return cls(target, sources, horizon_days)

    @property
    def feature_columns(self) -> list[str]:
        return [column for source in self.feature_sources for column in source.columns]

    @property
    def first_date(self) -> pd.Timestamp:
        return self.target.index[0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.target.index[-1]

    def features(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        """All feature columns for the given dates, as known on as_of."""
        frame = pd.DataFrame(index=dates)
        for source in self.feature_sources:
            frame = frame.join(source.features_for(dates, as_of))
        return frame

    def training_data(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Target and features for every date on or before as_of."""
        as_of = pd.Timestamp(as_of).normalize()
        history = self.target.loc[:as_of]
        return history.to_frame().join(self.features(history.index, as_of))

    def forecast_dates(self, as_of: pd.Timestamp) -> pd.DatetimeIndex:
        """The next horizon_days trading days after as_of.

        Inside the price history these are the real trading days. Past the end
        of the history they are business days.
        """
        as_of = pd.Timestamp(as_of).normalize()
        known = self.target.index[self.target.index > as_of][: self.horizon_days]
        missing = self.horizon_days - len(known)
        if missing == 0:
            return known
        start = known[-1] if len(known) else as_of
        extra = pd.bdate_range(start + pd.Timedelta(days=1), periods=missing)
        return known.append(extra)

    def window(self, as_of: pd.Timestamp) -> ForecastWindow:
        """Training data, future features, and actuals for one as-of date."""
        as_of = pd.Timestamp(as_of).normalize()
        dates = self.forecast_dates(as_of)
        return ForecastWindow(
            as_of=as_of,
            training_data=self.training_data(as_of),
            future_features=self.features(dates, as_of),
            test_target=self.target.reindex(dates),
        )

    def backtest_dates(
        self, start: pd.Timestamp, end: pd.Timestamp | None = None, step_days: int = 30
    ) -> list[pd.Timestamp]:
        """Rolling as-of dates from start to end, keeping only ones with a full test period."""
        end = pd.Timestamp(end) if end is not None else self.last_date
        candidates = pd.date_range(pd.Timestamp(start), end, freq=f"{step_days}D")
        dates = []
        for candidate in candidates:
            # Snap to the last trading day on or before the candidate
            history = self.target.index[self.target.index <= candidate]
            if len(history) == 0:
                continue
            as_of = history[-1]
            future = self.target.index[self.target.index > as_of]
            if len(future) >= self.horizon_days:
                dates.append(as_of)
        return sorted(set(dates))

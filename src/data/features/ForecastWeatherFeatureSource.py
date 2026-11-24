"""Degree days for live forecasts, where the future weather is not known yet."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from data.database.SqlReader import SqlReader
from data.features.WeatherFeatureSource import COLUMNS, WeatherFeatureSource

NORMAL_YEARS = 10
SMOOTHING_DAYS = 15
DAYS_IN_YEAR = 366


class ForecastWeatherFeatureSource(WeatherFeatureSource):
    """Weather for production scoring.

    Backtests use the observed weather for the forecast dates, which a live run
    cannot see. Here each date gets the best value known on the as-of date:

      - on or before the as-of date: observed degree days
      - after the as-of date, while the weather forecast lasts: the Open-Meteo
        forecast that the weather loader stores (about 14 days)
      - after that: the daily normal, the average degree days for that day of the
        year over the last 10 years of observed weather, smoothed over 15 days

    Only use this for the latest as-of date. Stored rows after an older as-of
    date are observed weather, not the forecast made back then.
    """

    def __init__(
        self,
        daily_weather: pd.DataFrame,
        normal_years: int = NORMAL_YEARS,
        smoothing_days: int = SMOOTHING_DAYS,
    ) -> None:
        super().__init__(daily_weather, use_observed_as_forecast=True)
        self.normal_years = normal_years
        self.smoothing_days = smoothing_days

    @classmethod
    def from_database(cls, engine: Engine) -> ForecastWeatherFeatureSource:
        query = text("SELECT date, location, hdd, cdd FROM weather_daily")
        frame = SqlReader.read(engine, query, ["date", "location", "hdd", "cdd"])
        return cls(frame)

    def normals(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Average degree days by day of year, from weather observed on or before as_of."""
        start = as_of - pd.DateOffset(years=self.normal_years)
        history = self.daily.loc[start:as_of]
        by_day = history.groupby(history.index.dayofyear)[COLUMNS].mean()
        by_day = by_day.reindex(range(1, DAYS_IN_YEAR + 1)).interpolate(limit_direction="both")
        # Wrap the year around so late December smooths with early January
        pad = self.smoothing_days
        wrapped = pd.concat([by_day.tail(pad), by_day, by_day.head(pad)])
        smoothed = wrapped.rolling(pad, center=True, min_periods=1).mean()
        return smoothed.iloc[pad:-pad]

    def kinds_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.Series:
        """Where each date's weather comes from: observed, forecast, or normal."""
        known = self.daily[COLUMNS].notna().all(axis=1).reindex(dates, fill_value=False)
        kinds = pd.Series("normal", index=dates)
        kinds[known] = "forecast"
        kinds[known & (dates <= as_of)] = "observed"
        return kinds

    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of).normalize()
        frame = self.daily.reindex(dates)
        missing = frame[COLUMNS].isna().any(axis=1) & (dates > as_of)
        if missing.any():
            normals = self.normals(as_of)
            days = dates[missing].dayofyear
            frame.loc[missing, COLUMNS] = normals.loc[days, COLUMNS].to_numpy()
        return frame

"""Heating and cooling degree days averaged across the configured cities."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from natgas.forecasting.data.AbstractFeatureSource import AbstractFeatureSource
from natgas.forecasting.data.SqlReader import SqlReader

COLUMNS = ["hdd", "cdd"]


class WeatherFeatureSource(AbstractFeatureSource):
    """Daily average heating and cooling degree days across all cities.

    For dates on or before the as-of date these are observed values.
    For dates after it, a real forecast would use the weather forecast made on
    the as-of date. Old weather forecasts are not stored, so backtests can either
    use the observed weather as a perfect forecast (use_observed_as_forecast=True)
    or leave future weather empty (False). A perfect forecast makes backtest
    scores a little better than live scores, so treat them as an upper bound.
    """

    def __init__(self, daily_weather: pd.DataFrame, use_observed_as_forecast: bool = True) -> None:
        frame = daily_weather.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        self.daily = frame.groupby("date")[COLUMNS].mean().sort_index()
        self.use_observed_as_forecast = use_observed_as_forecast

    @classmethod
    def from_database(
        cls, engine: Engine, use_observed_as_forecast: bool = True
    ) -> WeatherFeatureSource:
        query = text("SELECT date, location, hdd, cdd FROM weather_daily")
        frame = SqlReader.read(engine, query, ["date", "location", "hdd", "cdd"])
        return cls(frame, use_observed_as_forecast)

    @property
    def columns(self) -> list[str]:
        return list(COLUMNS)

    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        frame = self.daily.reindex(dates)
        if not self.use_observed_as_forecast:
            frame.loc[frame.index > as_of] = float("nan")
        return frame

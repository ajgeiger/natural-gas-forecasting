"""Fourier terms for the time of year and day of week, plus an after weekend flag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from natgas.forecasting.data.AbstractFeatureSource import AbstractFeatureSource

YEAR_DAYS = 365.25
WEEK_DAYS = 7
YEARLY_HARMONICS = 2
MAX_WEEKLY_HARMONICS = 2


class CalendarFeatureSource(AbstractFeatureSource):
    """Calendar features. They depend only on the date, so they are known ahead.

    year_sin_k, year_cos_k  sin and cos of 2 pi k day_of_year / 365.25, for k = 1, 2
    week_sin_k, week_cos_k  sin and cos of 2 pi k day_of_week / 7, for k = 1, 2
    after_weekend           1 when the day before was a Saturday or Sunday

    Prices are business daily, so there are no weekend rows. after_weekend
    marks the first weekday after a weekend instead, which is a Monday.
    """

    @staticmethod
    def column_names(
        use_yearly: bool = True,
        weekly_harmonics: int = MAX_WEEKLY_HARMONICS,
        use_weekend: bool = True,
    ) -> list[str]:
        """Column names for a choice of calendar features."""
        columns = []
        if use_yearly:
            for k in range(1, YEARLY_HARMONICS + 1):
                columns += [f"year_sin_{k}", f"year_cos_{k}"]
        for k in range(1, weekly_harmonics + 1):
            columns += [f"week_sin_{k}", f"week_cos_{k}"]
        if use_weekend:
            columns.append("after_weekend")
        return columns

    @property
    def columns(self) -> list[str]:
        return self.column_names()

    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        frame = pd.DataFrame(index=dates)
        day_of_year = dates.dayofyear.to_numpy()
        day_of_week = dates.dayofweek.to_numpy()
        for k in range(1, YEARLY_HARMONICS + 1):
            angle = 2 * np.pi * k * day_of_year / YEAR_DAYS
            frame[f"year_sin_{k}"] = np.sin(angle)
            frame[f"year_cos_{k}"] = np.cos(angle)
        for k in range(1, MAX_WEEKLY_HARMONICS + 1):
            angle = 2 * np.pi * k * day_of_week / WEEK_DAYS
            frame[f"week_sin_{k}"] = np.sin(angle)
            frame[f"week_cos_{k}"] = np.cos(angle)
        day_before = (dates - pd.Timedelta(days=1)).dayofweek.to_numpy()
        frame["after_weekend"] = (day_before >= 5).astype(float)
        return frame[self.columns]

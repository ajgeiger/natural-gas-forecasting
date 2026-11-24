"""US holiday dummies, plus the days before and after each holiday."""

from __future__ import annotations

import re

import holidays
import pandas as pd

from data.features.AbstractFeatureSource import AbstractFeatureSource

MAX_WINDOW = 4
FIRST_YEAR = 2000
LAST_YEAR = 2040
OBSERVED_SUFFIX = " (Observed)"


class HolidayFeatureSource(AbstractFeatureSource):
    """One dummy column per US holiday and per day around it.

    labor_day is 1 on Labor Day, labor_day_pre_2 is 1 two calendar days before
    it, and labor_day_post_1 is 1 the day after. window sets how many days
    before and after get a column, up to 4. An observed day counts as the
    holiday itself.

    Holidays are known years ahead, so values for the forecast dates are not
    look-ahead.
    """

    def __init__(self, window: int = MAX_WINDOW) -> None:
        if not 0 <= window <= MAX_WINDOW:
            raise ValueError(f"window must be between 0 and {MAX_WINDOW}")
        self.window = window
        days = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31")
        self.daily = pd.DataFrame(0.0, index=days, columns=self.columns)
        for day, name in self.holiday_days().items():
            for offset in range(-window, window + 1):
                column = self.column_for(name, offset)
                self.daily.at[day + pd.Timedelta(days=offset), column] = 1.0

    @staticmethod
    def holiday_days() -> dict[pd.Timestamp, str]:
        """Each holiday date with its snake_case name."""
        calendar = holidays.US(years=range(FIRST_YEAR, LAST_YEAR + 1))
        days = {}
        for day, name in calendar.items():
            name = name.replace(OBSERVED_SUFFIX, "").replace("'", "").lower()
            days[pd.Timestamp(day)] = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return days

    @staticmethod
    def column_for(name: str, offset: int) -> str:
        if offset < 0:
            return f"{name}_pre_{-offset}"
        if offset > 0:
            return f"{name}_post_{offset}"
        return name

    @classmethod
    def column_names(cls, window: int) -> list[str]:
        """Column names for a window, without building the dummies."""
        names = sorted(set(cls.holiday_days().values()))
        offsets = [0, *range(-1, -window - 1, -1), *range(1, window + 1)]
        return [cls.column_for(name, offset) for name in names for offset in offsets]

    @property
    def columns(self) -> list[str]:
        return self.column_names(self.window)

    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        return self.daily.reindex(dates, fill_value=0.0)

"""Trading days for NG=F, used to date forecasts past the end of the price history."""

from __future__ import annotations

import holidays
import pandas as pd
from dateutil.easter import easter

# US federal holidays the exchange stays open on
OPEN_HOLIDAYS = {"Columbus Day", "Veterans Day"}


class TradingCalendar:
    """Weekdays minus the US exchange holidays.

    The price history has no rows on US federal holidays (except Columbus Day
    and Veterans Day) or on Good Friday, so forecasts skip those days too.
    Otherwise a forecast for a closed day would never get an actual.
    """

    @staticmethod
    def holidays_for(years: range) -> set[pd.Timestamp]:
        closed = set()
        for day, name in holidays.US(years=years).items():
            if name.replace(" (Observed)", "") not in OPEN_HOLIDAYS:
                closed.add(pd.Timestamp(day))
        for year in years:
            closed.add(pd.Timestamp(easter(year)) - pd.Timedelta(days=2))
        return closed

    def next_days(self, after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
        """The first count trading days after the given date."""
        after = pd.Timestamp(after).normalize()
        # Two calendar days per trading day is always enough room
        candidates = pd.bdate_range(after + pd.Timedelta(days=1), periods=count * 2 + 10)
        closed = self.holidays_for(range(after.year, candidates[-1].year + 1))
        return candidates[~candidates.isin(list(closed))][:count]

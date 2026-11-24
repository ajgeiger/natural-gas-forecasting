"""Everything a model sees for one forecast date."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastWindow:
    """Training data up to the as-of date and the test period right after it.

    training_data: target column plus feature columns, dates on or before as_of.
    future_features: feature columns for the dates to predict, as known on as_of.
    test_target: actual target values for the dates to predict. Used only for scoring.
    """

    as_of: pd.Timestamp
    training_data: pd.DataFrame
    future_features: pd.DataFrame
    test_target: pd.Series

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        return self.future_features.index

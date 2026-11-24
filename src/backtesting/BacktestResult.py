"""Record of one backtest: one model, one as-of date."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """What a model predicted from one as-of date, what really happened, and the scores."""

    as_of: pd.Timestamp
    training_rows: int
    last_training_date: pd.Timestamp
    forecasts: pd.DataFrame
    metrics: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        """One row per forecast date, ready to stack with other backtests."""
        frame = self.forecasts.copy()
        frame.insert(0, "as_of", self.as_of)
        return frame.reset_index(names="date")

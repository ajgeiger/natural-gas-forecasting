"""What one production scoring run did."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class ScoringRun:
    run_id: str
    run_timestamp: datetime
    as_of_date: pd.Timestamp
    loaded: dict[str, int]
    saved: dict[str, int]
    actuals: int
    weather_days: dict[str, int]

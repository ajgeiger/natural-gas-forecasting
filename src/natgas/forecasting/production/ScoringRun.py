"""What one production scoring run did."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class ScoringRun:
    """saved is rows per stage. events maps each event to its winner, promoted lists
    the families whose challenger became champion, and stitched is the rows
    published to stage production."""

    run_id: str
    run_timestamp: datetime
    as_of_date: pd.Timestamp
    loaded: dict[str, int]
    saved: dict[str, int]
    actuals: int
    weather_days: dict[str, int]
    events: dict[str, str] = field(default_factory=dict)
    promoted: list[str] = field(default_factory=list)
    stitched: int = 0

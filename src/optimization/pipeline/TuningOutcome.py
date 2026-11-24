"""What a tuning run produced."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.BacktestReport import BacktestReport
from production.registry.PublishDecision import PublishDecision


@dataclass(frozen=True)
class TuningOutcome:
    """Results table, the best config per model, the publish decision, and the version."""

    summary: pd.DataFrame
    best_by_model: dict[str, BacktestReport]
    decision: PublishDecision
    registered_model_name: str
    registered_version: str | None
    parent_run_id: str

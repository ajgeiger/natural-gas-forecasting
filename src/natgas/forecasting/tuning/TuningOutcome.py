"""What a tuning run produced."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TuningOutcome:
    """Summary table, the winning config and its scores, and the registered version."""

    summary: pd.DataFrame
    best_model_name: str
    best_hyperparameters: dict
    best_metrics: dict
    registered_model_name: str
    registered_version: str | None
    parent_run_id: str

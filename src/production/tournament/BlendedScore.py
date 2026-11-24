"""One contender's scorecard in one horizon bucket."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlendedScore:
    """Production MAE over the recent window blended with the backtest MAE.

    prod_days       matured production as-of dates in the window, at most window_days
    prod_mae        MAE over those dates, 0.0 when there are none
    backtest_mae    backtest MAE of this version in this bucket
    backtest_source "model", or "naive" when the version had no backtest score here
    prod_weight     prod_days / window_days
    blended_mae     prod_weight * prod_mae + (1 - prod_weight) * backtest_mae
    """

    stage: str
    model: str
    model_version: str
    horizon_bucket: str
    prod_days: int
    prod_mae: float
    backtest_mae: float
    backtest_source: str
    prod_weight: float
    blended_mae: float

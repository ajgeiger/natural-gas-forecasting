"""Which model the publisher picked, and why."""

from __future__ import annotations

from dataclasses import dataclass

from natgas.forecasting.backtesting.BacktestReport import BacktestReport


@dataclass(frozen=True)
class PublishDecision:
    report: BacktestReport
    reason: str
    improvement_over_naive: float | None

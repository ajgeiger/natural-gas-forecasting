"""The result of one horizon event."""

from __future__ import annotations

from dataclasses import dataclass, field

from natgas.forecasting.production.tournament.BlendedScore import BlendedScore
from natgas.forecasting.production.tournament.Contender import Contender


@dataclass(frozen=True)
class EventResult:
    """Who won an event such as days 1 to 5, against what baseline, and why.

    entrants are the family champions with their scores and whether each
    qualified. The winner's forecasts fill first_day to last_day of the
    published forecast.
    """

    horizon_bucket: str
    first_day: int
    last_day: int
    winner: Contender
    winner_score: BlendedScore
    baseline_score: BlendedScore
    reason: str
    qualified: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Share of the baseline error the winner removed. 0.0 when naive wins."""
        return 1 - self.winner_score.blended_mae / self.baseline_score.blended_mae

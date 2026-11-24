"""What one tournament round decided."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from production.tournament.ChallengerTrial import ChallengerTrial
from production.tournament.EventResult import EventResult


@dataclass(frozen=True)
class TournamentOutcome:
    as_of_date: date
    events: list[EventResult] = field(default_factory=list)
    trials: list[ChallengerTrial] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    standings: int = 0
    stitched: int = 0

    def winners(self) -> dict[str, str]:
        """Event bucket to the winning stage and version, for logs and the CLI."""
        return {event.horizon_bucket: event.winner.label for event in self.events}

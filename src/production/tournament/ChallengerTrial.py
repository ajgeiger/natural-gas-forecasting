"""Where one family's challenger stands in its proving ground."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

IN_TRIALS = "in trials"
DEFENDED = "champion defended"
ELIGIBLE = "eligible"
PROMOTED = "promoted"


@dataclass(frozen=True)
class ChallengerTrial:
    """Trial progress on one as-of date.

    trial_days counts the as-of dates since the trial started whose forecasts
    have started to mature. champion_mae and challenger_mae are blended all
    days scores. The challenger is promoted once it has the required days
    and is ahead.
    """

    as_of_date: date
    family: str
    registered_model: str
    champion_version: str
    challenger_version: str
    trial_started: date | None
    trial_days: int
    required_days: int
    champion_mae: float
    challenger_mae: float
    promoted: bool = False
    promoted_at: datetime | None = None

    @property
    def challenger_ahead(self) -> bool:
        return self.challenger_mae < self.champion_mae

    @property
    def eligible(self) -> bool:
        return self.trial_days >= self.required_days and self.challenger_ahead

    @property
    def status(self) -> str:
        if self.promoted:
            return PROMOTED
        if self.trial_days < self.required_days:
            return IN_TRIALS
        return ELIGIBLE if self.challenger_ahead else DEFENDED

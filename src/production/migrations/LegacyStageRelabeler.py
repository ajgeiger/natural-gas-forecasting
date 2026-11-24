"""Moves forecasts from before the tournament out of the way of the new stages."""

from __future__ import annotations

from sqlalchemy import select, update

from data.database import EventWinner, ForecastActual, Prediction, session_scope

# Old stage name to the stage it keeps as history
LEGACY_STAGES = {"production": "legacy:champion", "staging": "legacy:challenger"}


class LegacyStageRelabeler:
    """Renames the stages of the single registry setup that came before the tournament.

    Before the tournament, "production" was the natgas-price-forecaster
    champion and "staging" its challenger. Now "production" means the
    stitched forecast of the event winners. Old rows keep their numbers and
    become "legacy:champion" and "legacy:challenger". Production rows of an
    as-of date that has event winners were written by the tournament and are
    left alone, so this is safe to run again.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def relabel(self) -> int:
        """Rows renamed across predictions and forecast_actuals."""
        tournament_dates = select(EventWinner.as_of_date).distinct()
        renamed = 0
        with session_scope(self.db_url) as session:
            for table in (Prediction, ForecastActual):
                for old, new in LEGACY_STAGES.items():
                    query = update(table).where(table.stage == old)
                    if old == "production":
                        query = query.where(table.as_of_date.not_in(tournament_dates))
                    renamed += session.execute(query.values(stage=new)).rowcount
        return renamed

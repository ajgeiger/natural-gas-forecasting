"""Empties the production and tournament tables before a replay."""

from __future__ import annotations

from sqlalchemy import delete, func, select

from natgas.db import (
    ChallengerTrialRecord,
    EventWinner,
    ForecastActual,
    Prediction,
    PromotionRecord,
    TournamentStanding,
    session_scope,
)

# Children first, so nothing refers to a deleted forecast
TABLES = {
    "forecast_actuals": ForecastActual,
    "tournament_standings": TournamentStanding,
    "event_winners": EventWinner,
    "challenger_trials": ChallengerTrialRecord,
    "promotion_history": PromotionRecord,
    "predictions": Prediction,
}


class ProductionHistoryReset:
    """Deletes every live forecast and tournament result.

    Source data (prices, weather, storage, ingestion runs) and the backtest
    forecasts are left alone, so a replay starts from a clean production
    history with the same inputs.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def clear(self) -> dict[str, int]:
        """Rows deleted per table."""
        removed = {}
        with session_scope(self.db_url) as session:
            for name, table in TABLES.items():
                removed[name] = int(
                    session.execute(select(func.count()).select_from(table)).scalar_one()
                )
                session.execute(delete(table))
        return removed

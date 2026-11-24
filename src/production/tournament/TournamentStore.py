"""Writes and reads the tournament tables."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import func, insert, select, text

from data.database import (
    ChallengerTrialRecord,
    EventWinner,
    PromotionRecord,
    TournamentStanding,
    get_engine,
    session_scope,
    upsert,
)
from data.database.SqlReader import SqlReader
from production.tournament.ChallengerTrial import ChallengerTrial

TABLES = {
    "tournament_standings": TournamentStanding,
    "event_winners": EventWinner,
    "challenger_trials": ChallengerTrialRecord,
    "promotion_history": PromotionRecord,
}


class TournamentStore:
    """tournament_standings, event_winners, challenger_trials, and promotion_history.

    Standings, winners, and trials have one row per as-of date and key, so a
    rerun for the same day replaces them. Promotions are only ever added.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def save_standings(self, rows: list[dict]) -> int:
        with session_scope(self.db_url) as session:
            return upsert(session, TournamentStanding, rows)

    def save_winners(self, rows: list[dict]) -> int:
        with session_scope(self.db_url) as session:
            return upsert(session, EventWinner, rows)

    def save_trial(self, trial: ChallengerTrial, run_id: str) -> int:
        row = {
            "as_of_date": trial.as_of_date,
            "family": trial.family,
            "run_id": run_id,
            "registered_model": trial.registered_model,
            "champion_version": trial.champion_version,
            "challenger_version": trial.challenger_version,
            "trial_started": trial.trial_started,
            "trial_days": trial.trial_days,
            "required_days": trial.required_days,
            "champion_mae": trial.champion_mae,
            "challenger_mae": trial.challenger_mae,
            "challenger_ahead": trial.challenger_ahead,
            "promoted": trial.promoted,
            "promoted_at": trial.promoted_at,
            "status": trial.status,
        }
        with session_scope(self.db_url) as session:
            return upsert(session, ChallengerTrialRecord, [row])

    def promotion_on(self, family: str, as_of_date: date) -> dict | None:
        """The promotion recorded for a family on an as-of date, if any."""
        query = select(PromotionRecord.__table__).where(
            PromotionRecord.family == family, PromotionRecord.as_of_date == as_of_date
        )
        with session_scope(self.db_url) as session:
            row = session.execute(query).mappings().first()
        return dict(row) if row else None

    def trial_on(self, family: str, as_of_date: date) -> dict | None:
        query = select(ChallengerTrialRecord.__table__).where(
            ChallengerTrialRecord.family == family, ChallengerTrialRecord.as_of_date == as_of_date
        )
        with session_scope(self.db_url) as session:
            row = session.execute(query).mappings().first()
        return dict(row) if row else None

    def record_promotion(self, trial: ChallengerTrial, run_id: str) -> None:
        row = {
            "as_of_date": trial.as_of_date,
            "family": trial.family,
            "registered_model": trial.registered_model,
            "new_champion_version": trial.challenger_version,
            "former_champion_version": trial.champion_version,
            "trial_days": trial.trial_days,
            "champion_mae": trial.champion_mae,
            "challenger_mae": trial.challenger_mae,
            "run_id": run_id,
        }
        if trial.promoted_at is not None:
            row["promoted_at"] = trial.promoted_at
        with session_scope(self.db_url) as session:
            session.execute(insert(PromotionRecord), [row])

    def has_winners(self, as_of_date: date) -> bool:
        query = select(func.count()).where(EventWinner.as_of_date == as_of_date)
        with session_scope(self.db_url) as session:
            return session.execute(query).scalar_one() > 0

    def latest(self, table: str) -> pd.DataFrame:
        """Rows of a tournament table for its newest as-of date."""
        model = TABLES[table]
        columns = [column.name for column in model.__table__.columns]
        query = text(
            f"SELECT {', '.join(columns)} FROM {table} "
            f"WHERE as_of_date = (SELECT MAX(as_of_date) FROM {table})"
        )
        return SqlReader.read(get_engine(self.db_url), query, columns)

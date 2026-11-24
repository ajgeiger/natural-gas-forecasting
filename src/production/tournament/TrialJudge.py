"""Judges each family's 60 day proving ground."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select

from data.database import ForecastActual, Prediction, PromotionRecord, session_scope
from production.tournament.BlendedScorer import BlendedScorer
from production.tournament.ChallengerTrial import ChallengerTrial
from production.tournament.Contender import Contender
from production.tournament.TournamentConfig import ALL_DAYS


class TrialJudge:
    """Counts a challenger's matured production days and compares it with its champion.

    The trial starts on the first as-of date the current challenger version
    was scored as challenger after the family's last promotion. A trial day is
    an as-of date in that run whose forecasts have started to mature. Both
    models are compared on the blended all days MAE, the same scorecard the
    events use. Promotion itself is left to the caller.
    """

    def __init__(self, db_url: str, scorer: BlendedScorer, required_days: int = 60) -> None:
        self.db_url = db_url
        self.scorer = scorer
        self.required_days = required_days

    def last_promotion(self, family: str, as_of_date: date) -> date | None:
        query = select(func.max(PromotionRecord.as_of_date)).where(
            PromotionRecord.family == family, PromotionRecord.as_of_date <= as_of_date
        )
        with session_scope(self.db_url) as session:
            return session.execute(query).scalar_one()

    def trial_started(self, challenger: Contender, as_of_date: date) -> date | None:
        query = select(func.min(Prediction.as_of_date)).where(
            Prediction.stage == challenger.stage,
            Prediction.model_version == challenger.version,
            Prediction.as_of_date <= as_of_date,
        )
        promoted = self.last_promotion(challenger.family, as_of_date)
        if promoted is not None:
            query = query.where(Prediction.as_of_date > promoted)
        with session_scope(self.db_url) as session:
            return session.execute(query).scalar_one()

    def trial_days(self, challenger: Contender, started: date | None, as_of_date: date) -> int:
        if started is None:
            return 0
        query = select(func.count(func.distinct(ForecastActual.as_of_date))).where(
            ForecastActual.stage == challenger.stage,
            ForecastActual.model_version == challenger.version,
            ForecastActual.as_of_date >= started,
            ForecastActual.as_of_date <= as_of_date,
        )
        with session_scope(self.db_url) as session:
            return int(session.execute(query).scalar_one())

    def judge(
        self,
        champion: Contender,
        challenger: Contender,
        as_of_date: date,
        fallback_mae: dict[str, float] | None = None,
    ) -> ChallengerTrial:
        started = self.trial_started(challenger, as_of_date)
        champion_score = self.scorer.score(champion, ALL_DAYS, as_of_date, fallback_mae)
        challenger_score = self.scorer.score(challenger, ALL_DAYS, as_of_date, fallback_mae)
        return ChallengerTrial(
            as_of_date=as_of_date,
            family=champion.family,
            registered_model=champion.registered_model,
            champion_version=str(champion.version),
            challenger_version=str(challenger.version),
            trial_started=started,
            trial_days=self.trial_days(challenger, started, as_of_date),
            required_days=self.required_days,
            champion_mae=champion_score.blended_mae,
            challenger_mae=challenger_score.blended_mae,
        )

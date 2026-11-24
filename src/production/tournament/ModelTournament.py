"""One daily round of the tournament: trials, events, and the published forecast."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime

from data.database.models import utc_now
from production.scoring.PredictionStore import PredictionStore
from production.tournament.BlendedScore import BlendedScore
from production.tournament.BlendedScorer import BlendedScorer
from production.tournament.ChallengerTrial import ChallengerTrial
from production.tournament.Contender import (
    BASELINE,
    CHALLENGER,
    CHAMPION,
    Contender,
)
from production.tournament.EventJudge import EventJudge
from production.tournament.EventResult import EventResult
from production.tournament.FamilyRegistry import FamilyRegistry
from production.tournament.ForecastStitcher import ForecastStitcher
from production.tournament.ModelRules import ModelRules
from production.tournament.TournamentConfig import (
    EVENTS,
    SCORED_BUCKETS,
    TournamentConfig,
)
from production.tournament.TournamentOutcome import TournamentOutcome
from production.tournament.TournamentStore import TournamentStore
from production.tournament.TrialJudge import TrialJudge

log = logging.getLogger(__name__)


class ModelTournament:
    """Runs after the day's forecasts are scored and the matured actuals are in.

    1. Trials: each family's challenger is judged against its champion. With
       60 matured production days and a lower blended MAE it is promoted.
    2. Scorecard: every contender gets a blended MAE in each event and over
       all days.
    3. Events: family champions that follow the rules and beat naive by the
       margin qualify, and the lowest blended MAE wins. Naive wins an event
       nobody qualifies for.
    4. Publish: the three winners are stitched into stage production.

    Every step is saved, so the dashboard can show the standings, the
    winners, the trial progress, and the promotions.
    """

    def __init__(
        self, config: TournamentConfig, db_url: str, registry: FamilyRegistry | None = None
    ) -> None:
        self.config = config
        self.db_url = db_url
        self.registry = registry or FamilyRegistry(config)
        self.scorer = BlendedScorer(db_url, config.window_days)
        self.store = TournamentStore(db_url)
        self.event_judge = EventJudge(config.min_improvement)

    @staticmethod
    def by_stage(contenders: list[Contender]) -> dict[str, Contender]:
        return {contender.stage: contender for contender in contenders}

    def run_trials(
        self, contenders: list[Contender], as_of_date: date, run_id: str, fallback: dict
    ) -> tuple[list[ChallengerTrial], list[str]]:
        """Judge every family's trial and promote the challengers that earned it."""
        judge = TrialJudge(self.db_url, self.scorer, self.config.trial_days)
        stages = self.by_stage(contenders)
        trials, promoted = [], []
        for family in self.config.families:
            if self.store.promotion_on(family, as_of_date):
                log.info("%s already promoted on %s, trial kept as recorded", family, as_of_date)
                continue
            champion = stages[f"{family}:{CHAMPION}"]
            challenger = stages[f"{family}:{CHALLENGER}"]
            trial = judge.judge(champion, challenger, as_of_date, fallback)
            if trial.eligible:
                self.registry.promote(family)
                trial = replace(trial, promoted=True, promoted_at=utc_now())
                self.store.record_promotion(trial, run_id)
                promoted.append(family)
                log.info(
                    "%s challenger v%s promoted to champion after %d days",
                    family,
                    trial.challenger_version,
                    trial.trial_days,
                )
            self.store.save_trial(trial, run_id)
            trials.append(trial)
        return trials, promoted

    def score_all(
        self, contenders: list[Contender], as_of_date: date, fallback: dict
    ) -> dict[tuple[str, str], BlendedScore]:
        return {
            (contender.stage, bucket): self.scorer.score(contender, bucket, as_of_date, fallback)
            for contender in contenders
            for bucket in SCORED_BUCKETS
        }

    def run_events(
        self, contenders: list[Contender], scores: dict[tuple[str, str], BlendedScore]
    ) -> list[EventResult]:
        champions = [contender for contender in contenders if contender.role == CHAMPION]
        baseline = next(contender for contender in contenders if contender.role == BASELINE)
        events = []
        for first, last, bucket in EVENTS:
            entrants = [(champion, scores[(champion.stage, bucket)]) for champion in champions]
            event = self.event_judge.judge(
                bucket, first, last, entrants, (baseline, scores[(baseline.stage, bucket)])
            )
            log.info("%s: %s", bucket, event.reason)
            events.append(event)
        return events

    @staticmethod
    def standing_rows(
        contenders: list[Contender],
        scores: dict[tuple[str, str], BlendedScore],
        events: list[EventResult],
        as_of_date: date,
        run_id: str,
    ) -> list[dict]:
        baseline = next(contender for contender in contenders if contender.role == BASELINE)
        by_bucket = {event.horizon_bucket: event for event in events}
        rows = []
        for bucket in SCORED_BUCKETS:
            ranked = sorted(contenders, key=lambda item: scores[(item.stage, bucket)].blended_mae)
            baseline_mae = scores[(baseline.stage, bucket)].blended_mae
            event = by_bucket.get(bucket)
            for rank, contender in enumerate(ranked, start=1):
                score = scores[(contender.stage, bucket)]
                in_event = event is not None and contender.role in (CHAMPION, BASELINE)
                qualified = in_event and (
                    contender.role == BASELINE or contender.stage in event.qualified
                )
                rows.append(
                    {
                        "as_of_date": as_of_date,
                        "horizon_bucket": bucket,
                        "stage": contender.stage,
                        "run_id": run_id,
                        "family": contender.family,
                        "role": contender.role,
                        "model": contender.model,
                        "model_version": str(contender.version),
                        "registered_model": contender.registered_model,
                        "prod_days": score.prod_days,
                        "prod_mae": score.prod_mae,
                        "backtest_mae": score.backtest_mae,
                        "backtest_source": score.backtest_source,
                        "prod_weight": score.prod_weight,
                        "blended_mae": score.blended_mae,
                        "baseline_mae": baseline_mae,
                        "improvement": 1 - score.blended_mae / baseline_mae,
                        "rules_passed": ModelRules.passes(contender.config),
                        "qualified": qualified,
                        "rank": rank,
                        "is_winner": event is not None and event.winner.stage == contender.stage,
                        "created_at": utc_now(),
                    }
                )
        return rows

    @staticmethod
    def winner_rows(events: list[EventResult], as_of_date: date, run_id: str) -> list[dict]:
        return [
            {
                "as_of_date": as_of_date,
                "horizon_bucket": event.horizon_bucket,
                "run_id": run_id,
                "first_day": event.first_day,
                "last_day": event.last_day,
                "family": event.winner.family,
                "stage": event.winner.stage,
                "model": event.winner.model,
                "model_version": str(event.winner.version),
                "registered_model": event.winner.registered_model,
                "blended_mae": event.winner_score.blended_mae,
                "baseline_mae": event.baseline_score.blended_mae,
                "improvement": event.improvement,
                "qualified_count": len(event.qualified),
                "reason": event.reason,
                "created_at": utc_now(),
            }
            for event in events
        ]

    def run(self, as_of_date: date, run_id: str, run_timestamp: datetime) -> TournamentOutcome:
        contenders = self.registry.load_all()
        baseline = next(contender for contender in contenders if contender.role == BASELINE)
        fallback = baseline.backtest_mae

        trials, promoted = self.run_trials(contenders, as_of_date, run_id, fallback)
        if promoted:
            contenders = self.registry.load_all()

        scores = self.score_all(contenders, as_of_date, fallback)
        events = self.run_events(contenders, scores)
        standings = self.store.save_standings(
            self.standing_rows(contenders, scores, events, as_of_date, run_id)
        )
        self.store.save_winners(self.winner_rows(events, as_of_date, run_id))
        stitcher = ForecastStitcher(PredictionStore(self.db_url))
        stitched = stitcher.stitch(as_of_date, events, baseline, run_id, run_timestamp)
        return TournamentOutcome(as_of_date, events, trials, promoted, standings, stitched)

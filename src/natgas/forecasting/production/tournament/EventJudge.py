"""Runs one horizon event between the family champions."""

from __future__ import annotations

from natgas.forecasting.production.tournament.BlendedScore import BlendedScore
from natgas.forecasting.production.tournament.Contender import Contender
from natgas.forecasting.production.tournament.EventResult import EventResult
from natgas.forecasting.production.tournament.ModelRules import ModelRules


class EventJudge:
    """Qualifies the champions for an event and picks the winner.

    A champion qualifies when it follows the domain rules and its blended MAE
    in the event beats the baseline by at least min_improvement (0.01 means
    1 percent lower error). The qualified champion with the lowest blended MAE
    wins. When nobody qualifies the baseline wins, because a model that does
    not clearly beat "repeat the last price" is not worth publishing.
    """

    def __init__(self, min_improvement: float = 0.01) -> None:
        self.min_improvement = min_improvement

    def qualifies(self, champion: Contender, score: BlendedScore, baseline: BlendedScore) -> bool:
        if champion.config is None or not ModelRules.passes(champion.config):
            return False
        return score.blended_mae <= baseline.blended_mae * (1 - self.min_improvement)

    def judge(
        self,
        bucket: str,
        first_day: int,
        last_day: int,
        champions: list[tuple[Contender, BlendedScore]],
        baseline: tuple[Contender, BlendedScore],
    ) -> EventResult:
        baseline_contender, baseline_score = baseline
        qualified = [
            (champion, score)
            for champion, score in champions
            if self.qualifies(champion, score, baseline_score)
        ]
        if not qualified:
            reason = (
                f"no champion beat {baseline_contender.family} by "
                f"{self.min_improvement:.2%}, so {baseline_contender.family} is published"
            )
            return EventResult(
                bucket,
                first_day,
                last_day,
                baseline_contender,
                baseline_score,
                baseline_score,
                reason,
            )

        winner, winner_score = min(qualified, key=lambda pair: pair[1].blended_mae)
        improvement = 1 - winner_score.blended_mae / baseline_score.blended_mae
        reason = (
            f"{winner.family} won {bucket} with blended MAE {winner_score.blended_mae:.4f}, "
            f"{improvement:.2%} better than {baseline_contender.family}"
        )
        return EventResult(
            bucket,
            first_day,
            last_day,
            winner,
            winner_score,
            baseline_score,
            reason,
            [champion.stage for champion, _ in qualified],
        )

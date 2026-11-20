"""Builds the business facing forecast from the three event winners."""

from __future__ import annotations

import logging
from datetime import date, datetime

from natgas.forecasting.production.PredictionStore import PredictionStore
from natgas.forecasting.production.tournament.BlendedScorer import BlendedScorer
from natgas.forecasting.production.tournament.Contender import Contender
from natgas.forecasting.production.tournament.EventResult import EventResult

log = logging.getLogger(__name__)

PRODUCTION_STAGE = "production"


class ForecastStitcher:
    """Copies each event winner's forecasts for its days into stage "production".

    Days 1 to 5 come from the days 1 to 5 winner, and so on, so the business
    sees one 30 day forecast. Each row keeps the winner's model and version.
    The rows are copies of what the winner forecast in its own stage today,
    so the published numbers are exactly the ones that were scored. If a
    winner has no forecast for a day the baseline fills it. Like every stage,
    production is written once per as-of date.
    """

    def __init__(self, store: PredictionStore) -> None:
        self.store = store

    @staticmethod
    def pick(rows: list[dict], contender: Contender, horizon_day: int) -> dict | None:
        stages = BlendedScorer.stages_for(contender)
        for row in rows:
            if (
                row["stage"] in stages
                and row["model"] == contender.model
                and row["model_version"] == contender.version
                and row["horizon_day"] == horizon_day
            ):
                return row
        return None

    def stitch(
        self,
        as_of_date: date,
        events: list[EventResult],
        baseline: Contender,
        run_id: str,
        run_timestamp: datetime,
    ) -> int:
        """Rows written to stage production. Zero when it already has this as-of date."""
        if self.store.has_run(as_of_date, PRODUCTION_STAGE):
            log.info("production already published for %s, skipping", as_of_date)
            return 0
        stages = {stage for event in events for stage in BlendedScorer.stages_for(event.winner)}
        stages.add(baseline.stage)
        rows = self.store.rows(as_of_date, sorted(stages))

        published = []
        for event in events:
            for horizon_day in range(event.first_day, event.last_day + 1):
                row = self.pick(rows, event.winner, horizon_day)
                if row is None:
                    log.warning(
                        "%s has no day %d, using the baseline", event.winner.label, horizon_day
                    )
                    row = self.pick(rows, baseline, horizon_day)
                if row is None:
                    continue
                published.append(
                    {
                        **row,
                        "stage": PRODUCTION_STAGE,
                        "run_id": run_id,
                        "run_timestamp": run_timestamp,
                    }
                )
        return self.store.append(published)

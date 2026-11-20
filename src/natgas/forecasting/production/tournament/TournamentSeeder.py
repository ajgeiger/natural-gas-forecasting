"""Seeds every family registry from the latest grid search."""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
from mlflow.entities import Run

from natgas.forecasting.backtesting.ForecastMetrics import ForecastMetrics
from natgas.forecasting.data.ForecastDataProvider import ForecastDataProvider
from natgas.forecasting.production.GridSearchRuns import GridSearchRuns
from natgas.forecasting.production.tournament.Contender import BASELINE, CHALLENGER, CHAMPION
from natgas.forecasting.production.tournament.FamilyRegistry import FamilyRegistry
from natgas.forecasting.production.tournament.ModelRules import ModelRules
from natgas.forecasting.production.tournament.SeededVersion import SeededVersion
from natgas.forecasting.production.tournament.TournamentConfig import TournamentConfig
from natgas.forecasting.tuning.ModelPublisher import ModelPublisher

log = logging.getLogger(__name__)

# Two scores this close come from configs that forecast the same thing
SAME_SCORE = 1e-12


class TournamentSeeder:
    """Opens the tournament: each family gets a champion and a challenger.

    From the latest grid search, the best config of each family that follows
    the domain rules becomes @champion and the next best distinct config
    becomes @challenger. A config is distinct when its score differs too,
    since some params change nothing (Greykite's holiday window with holiday
    dummies off). The baseline is registered under @baseline.

    Configs are not backtested again. Each one is retrained on data through
    as_of and published with the scores the grid search logged, including the
    MAE of each horizon bucket. Seeding stops if a family already has a
    champion, unless force is set, so trial progress is never wiped by
    accident.
    """

    def __init__(
        self,
        runs: GridSearchRuns,
        provider: ForecastDataProvider,
        config: TournamentConfig,
        as_of: pd.Timestamp | None = None,
    ) -> None:
        self.runs = runs
        self.provider = provider
        self.config = config
        self.as_of = as_of
        self.registry = FamilyRegistry(config)

    @staticmethod
    def summary_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Overall and per bucket scores, without the per step ones."""
        return {
            name: value
            for name, value in metrics.items()
            if name in ForecastMetrics.names
            or any(name.startswith(f"{metric}_days_") for metric in ForecastMetrics.names)
        }

    def pick(self, runs: list[Run], count: int = 2) -> list[Run]:
        """The best `count` distinct configs that follow the rules, best first."""
        picked: list[Run] = []
        metric = self.runs.selection_metric
        for run in runs:
            config = self.runs.config_for(run)
            if not ModelRules.passes(config):
                continue
            score = run.data.metrics[metric]
            if any(abs(score - chosen.data.metrics[metric]) < SAME_SCORE for chosen in picked):
                continue
            picked.append(run)
            if len(picked) == count:
                break
        return picked

    def plan(self) -> tuple[Run, list[tuple[str, str, Run]]]:
        """The search and (family, role, run) for every title."""
        search = self.runs.latest_search()
        ranked = self.runs.ranked_runs(search)
        plan = []
        for family in self.config.families:
            picked = self.pick(ranked.get(family, []))
            if len(picked) < 2:
                raise ValueError(f"{family} needs two valid configs in the grid search")
            plan += [(family, CHAMPION, picked[0]), (family, CHALLENGER, picked[1])]
        baseline = ranked.get(self.config.baseline)
        if not baseline:
            raise ValueError(f"the grid search has no {self.config.baseline} run")
        plan.append((self.config.baseline, BASELINE, baseline[0]))
        return search, plan

    def seed(self, force: bool = False) -> list[SeededVersion]:
        if not force:
            seeded = [
                family
                for family in self.config.families
                if self.registry.version_for(self.config.registered_model(family), CHAMPION)
            ]
            if seeded:
                raise ValueError(f"{seeded} already have a champion. Use force to seed again")

        search, plan = self.plan()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.runs.experiment_name)
        metric = self.runs.selection_metric
        seeded = []
        for family, role, run in plan:
            name = self.config.registered_model(family)
            publisher = ModelPublisher(self.provider, name, role, metric)
            config = self.runs.config_for(run)
            reason = f"{run.info.run_name} from grid search {search.info.run_id}, seeded as {role}"
            with mlflow.start_run(run_name=f"seed-{family}-{role}"):
                mlflow.set_tags(
                    {
                        "search_run_id": search.info.run_id,
                        "source_run_id": run.info.run_id,
                        "method": "tournament-seed",
                        "family": family,
                        "role": role,
                    }
                )
                metrics = self.summary_metrics(run.data.metrics)
                version = publisher.publish_config(config, metrics, reason, self.as_of)
            seeded.append(
                SeededVersion(
                    family, role, name, version, run.info.run_name, run.data.metrics[metric]
                )
            )
            log.info("%s@%s is version %s (%s)", name, role, version, run.info.run_name)
        return seeded

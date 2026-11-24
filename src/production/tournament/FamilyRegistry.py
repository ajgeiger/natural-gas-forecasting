"""The per family MLflow registries: who holds each title, and promotions."""

from __future__ import annotations

from dataclasses import replace

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from production.registry.RegisteredModelLoader import RegisteredModelLoader
from production.tournament.Contender import (
    BASELINE,
    CHALLENGER,
    CHAMPION,
    Contender,
)
from production.tournament.TournamentConfig import (
    SCORED_BUCKETS,
    TournamentConfig,
)

LEGACY_ALIAS = "legacy_champion"


class FamilyRegistry:
    """Each family has its own registered model, natgas-chronos for example.

    Its @champion alias is the family title holder and @challenger is the
    model in trials. The baseline has its own registered model with a
    @baseline alias. Every version carries its backtest scores as tags, such
    as backtest_mae_days_1_to_5, written when it was published.
    """

    def __init__(self, config: TournamentConfig) -> None:
        self.config = config
        self.client = MlflowClient(config.tracking_uri)

    def slots(self) -> list[Contender]:
        """Every contender to score: champion and challenger per family, then the baseline."""
        contenders = [
            Contender(family, role, self.config.registered_model(family))
            for family in self.config.families
            for role in (CHAMPION, CHALLENGER)
        ]
        baseline = self.config.baseline
        contenders.append(Contender(baseline, BASELINE, self.config.registered_model(baseline)))
        return contenders

    def version_for(self, registered_model: str, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(registered_model, alias).version)
        except MlflowException:
            return None

    def backtest_mae(self, registered_model: str, version: str) -> dict[str, float]:
        """Backtest MAE per bucket from the version tags. Missing buckets are left out."""
        tags = self.client.get_model_version(registered_model, version).tags
        scores = {}
        for bucket in SCORED_BUCKETS:
            value = tags.get(f"backtest_{TournamentConfig.metric_name(bucket)}")
            if value is not None:
                scores[bucket] = float(value)
        return scores

    def load(self, contender: Contender) -> Contender:
        """The contender with its version, config, and backtest MAE filled in."""
        loader = RegisteredModelLoader(self.config.tracking_uri, contender.registered_model)
        version, config = loader.load(contender.alias)
        backtest = self.backtest_mae(contender.registered_model, version)
        return replace(contender, version=version, config=config, backtest_mae=backtest)

    def load_all(self) -> list[Contender]:
        missing = [
            f"{slot.registered_model}@{slot.alias}"
            for slot in self.slots()
            if self.version_for(slot.registered_model, slot.alias) is None
        ]
        if missing:
            raise ValueError(
                f"Tournament is not seeded, missing {missing}. Run natgas tournament seed"
            )
        return [self.load(slot) for slot in self.slots()]

    def promote(self, family: str) -> tuple[str, str]:
        """Swap the titles: the challenger becomes champion and the former champion
        goes back to trials as the new challenger, also tagged @legacy_champion.

        Returns the new champion version and the former champion version.
        """
        name = self.config.registered_model(family)
        champion = self.version_for(name, CHAMPION)
        challenger = self.version_for(name, CHALLENGER)
        if champion is None or challenger is None:
            raise ValueError(f"{name} needs both a champion and a challenger to promote")
        self.client.set_registered_model_alias(name, CHAMPION, challenger)
        self.client.set_registered_model_alias(name, CHALLENGER, champion)
        self.client.set_registered_model_alias(name, LEGACY_ALIAS, champion)
        return challenger, champion

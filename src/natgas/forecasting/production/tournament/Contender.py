"""One model in the tournament: a family champion, a family challenger, or the baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

CHAMPION = "champion"
CHALLENGER = "challenger"
BASELINE = "baseline"
ROLES = [CHAMPION, CHALLENGER]


@dataclass(frozen=True)
class Contender:
    """A registry alias and, once loaded, the version, config, and backtest MAE behind it.

    The role is also the registry alias. The stage is the name its forecasts
    get in the predictions table, for example "chronos:champion" or
    "baseline:naive".
    """

    family: str
    role: str
    registered_model: str
    version: str | None = None
    config: AbstractConfigWrapper | None = None
    backtest_mae: dict[str, float] = field(default_factory=dict)

    @property
    def alias(self) -> str:
        return self.role

    @property
    def stage(self) -> str:
        if self.role == BASELINE:
            return f"{BASELINE}:{self.family}"
        return f"{self.family}:{self.role}"

    @property
    def model(self) -> str:
        """Model name in the predictions table, the same as the family."""
        return self.config.model_name if self.config is not None else self.family

    @property
    def label(self) -> str:
        return f"{self.stage} v{self.version}"

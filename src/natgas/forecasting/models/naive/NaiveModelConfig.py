"""Settings for the naive baseline. It has no hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper


@dataclass(frozen=True)
class NaiveModelConfig(AbstractConfigWrapper):
    model_name = "naive"

    def validate(self) -> None:
        return None

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {}

    @classmethod
    def default_search_space(cls) -> dict:
        return {}

    @property
    def min_training_rows(self) -> int:
        return 1

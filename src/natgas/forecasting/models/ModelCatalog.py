"""Looks up model classes by name, so configs can refer to models as text."""

from __future__ import annotations

from pathlib import Path

from natgas.forecasting.models.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.GradientBoostingModel import GradientBoostingModel
from natgas.forecasting.models.NaiveModel import NaiveModel


class ModelCatalog:
    """Maps names like "naive" in tuning.yaml to model classes.

    Add a new model here and it can be tuned, backtested, and published.
    """

    models: dict[str, type[AbstractModelWrapper]] = {
        NaiveModel.name: NaiveModel,
        GradientBoostingModel.name: GradientBoostingModel,
    }

    @classmethod
    def get(cls, name: str) -> type[AbstractModelWrapper]:
        if name not in cls.models:
            raise KeyError(f"Unknown model {name!r}. Choose from {sorted(cls.models)}")
        return cls.models[name]

    @classmethod
    def load(cls, folder: str | Path) -> AbstractModelWrapper:
        """Load any saved model by reading which class wrote it."""
        name = AbstractModelWrapper.read_settings(folder)["name"]
        return cls.get(name).load(folder)

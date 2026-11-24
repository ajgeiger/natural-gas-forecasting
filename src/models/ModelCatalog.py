"""Looks up model and config classes by name, so YAML can refer to models as text."""

from __future__ import annotations

from pathlib import Path

from models.chronos.ChronosModel import ChronosModel
from models.chronos.ChronosModelConfig import ChronosModelConfig
from models.config.AbstractConfigWrapper import AbstractConfigWrapper
from models.config.AbstractModelWrapper import AbstractModelWrapper
from models.gradient_boosting.GradientBoostingModel import (
    GradientBoostingModel,
)
from models.gradient_boosting.GradientBoostingModelConfig import (
    GradientBoostingModelConfig,
)
from models.greykite.GreykiteModel import GreykiteModel
from models.greykite.GreykiteModelConfig import GreykiteModelConfig
from models.naive.NaiveModel import NaiveModel
from models.naive.NaiveModelConfig import NaiveModelConfig


class ModelCatalog:
    """Maps a name like "greykite" to its (model class, config class) pair.

    Add a new model here and it can be backtested, tuned, and published.
    """

    models: dict[str, tuple[type[AbstractModelWrapper], type[AbstractConfigWrapper]]] = {
        NaiveModelConfig.model_name: (NaiveModel, NaiveModelConfig),
        GradientBoostingModelConfig.model_name: (
            GradientBoostingModel,
            GradientBoostingModelConfig,
        ),
        GreykiteModelConfig.model_name: (GreykiteModel, GreykiteModelConfig),
        ChronosModelConfig.model_name: (ChronosModel, ChronosModelConfig),
    }

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.models)

    @classmethod
    def entry(cls, name: str) -> tuple[type[AbstractModelWrapper], type[AbstractConfigWrapper]]:
        if name not in cls.models:
            raise KeyError(f"Unknown model {name!r}. Choose from {cls.names()}")
        return cls.models[name]

    @classmethod
    def model_class(cls, name: str) -> type[AbstractModelWrapper]:
        return cls.entry(name)[0]

    @classmethod
    def config_class(cls, name: str) -> type[AbstractConfigWrapper]:
        return cls.entry(name)[1]

    @classmethod
    def build(cls, config: AbstractConfigWrapper) -> AbstractModelWrapper:
        """A fresh, untrained model for a config."""
        return cls.model_class(config.model_name)(config)

    @classmethod
    def load(cls, folder: str | Path) -> AbstractModelWrapper:
        """Load any saved model by reading which model wrote it."""
        name = AbstractModelWrapper.read_settings(folder)["name"]
        return cls.model_class(name).load(folder)

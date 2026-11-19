"""Forecasting models. Each one has a model class and a matching config class."""

from natgas.forecasting.models.base import AbstractConfigWrapper, AbstractModelWrapper
from natgas.forecasting.models.chronos import ChronosModel, ChronosModelConfig
from natgas.forecasting.models.gradient_boosting import (
    GradientBoostingModel,
    GradientBoostingModelConfig,
)
from natgas.forecasting.models.greykite import GreykiteModel, GreykiteModelConfig
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.models.naive import NaiveModel, NaiveModelConfig

__all__ = [
    "AbstractConfigWrapper",
    "AbstractModelWrapper",
    "ChronosModel",
    "ChronosModelConfig",
    "GradientBoostingModel",
    "GradientBoostingModelConfig",
    "GreykiteModel",
    "GreykiteModelConfig",
    "ModelCatalog",
    "NaiveModel",
    "NaiveModelConfig",
]

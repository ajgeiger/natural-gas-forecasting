"""Forecasting models that share one interface."""

from natgas.forecasting.models.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.GradientBoostingModel import GradientBoostingModel
from natgas.forecasting.models.ModelCatalog import ModelCatalog
from natgas.forecasting.models.NaiveModel import NaiveModel

__all__ = ["AbstractModelWrapper", "GradientBoostingModel", "ModelCatalog", "NaiveModel"]

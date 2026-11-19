"""Shared interface for every forecasting model."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

SETTINGS_FILE = "model.json"


class AbstractModelWrapper(ABC):
    """Every model trains, predicts, saves, and loads the same way.

    The constructor takes the model's config object. train() gets the training
    data from a ForecastWindow: a "price" column plus feature columns, indexed
    by date. predict() gets the feature values for the dates to forecast and
    returns one prediction per date. Because every model follows this contract,
    the backtester, optimizers, and publisher work with any of them.
    """

    config_class: ClassVar[type[AbstractConfigWrapper]]

    def __init__(self, config: AbstractConfigWrapper | None = None) -> None:
        config = config if config is not None else self.config_class()
        if not isinstance(config, self.config_class):
            raise TypeError(f"{type(self).__name__} needs a {self.config_class.__name__}")
        self.config = config
        self.is_trained = False

    @property
    def name(self) -> str:
        return self.config.model_name

    @abstractmethod
    def train(self, training_data: pd.DataFrame) -> None:
        """Fit the model on data up to and including the as-of date."""

    @abstractmethod
    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        """Predict the target for each date in future_features.index."""

    @abstractmethod
    def save(self, folder: str | Path) -> None:
        """Write everything needed to rebuild the trained model into folder."""

    @classmethod
    @abstractmethod
    def load(cls, folder: str | Path) -> AbstractModelWrapper:
        """Rebuild a trained model from a folder written by save()."""

    def write_settings(self, folder: str | Path, extra: dict | None = None) -> Path:
        """Save the model name and config. Subclasses call this from save()."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        settings = {"name": self.name, "config": self.config.to_dict(), **(extra or {})}
        (folder / SETTINGS_FILE).write_text(json.dumps(settings, indent=2, default=str))
        return folder

    @staticmethod
    def read_settings(folder: str | Path) -> dict:
        return json.loads((Path(folder) / SETTINGS_FILE).read_text())

    @classmethod
    def config_from_folder(cls, folder: str | Path) -> AbstractConfigWrapper:
        return cls.config_class.from_dict(cls.read_settings(folder)["config"])

    def check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(f"{self.name} must be trained before it can predict or save")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config.to_dict()})"

"""Shared interface for every forecasting model."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

SETTINGS_FILE = "model.json"


class AbstractModelWrapper(ABC):
    """Every model trains, predicts, saves, and loads the same way.

    train() gets the training data from a ForecastWindow: a "price" column plus
    feature columns, indexed by date. predict() gets the feature values for the
    dates to forecast and returns one prediction per date. Because all models
    follow this contract, the backtester and the tuner work with any of them.
    """

    name = ""
    default_hyperparameters: dict = {}

    def __init__(self, hyperparameters: dict | None = None) -> None:
        unknown = set(hyperparameters or {}) - set(self.default_hyperparameters)
        if unknown:
            raise ValueError(f"{self.name} does not accept: {sorted(unknown)}")
        self.hyperparameters = {**self.default_hyperparameters, **(hyperparameters or {})}
        self.is_trained = False

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
        """Save the model name and hyperparameters. Subclasses call this from save()."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        settings = {"name": self.name, "hyperparameters": self.hyperparameters, **(extra or {})}
        (folder / SETTINGS_FILE).write_text(json.dumps(settings, indent=2, default=str))
        return folder

    @staticmethod
    def read_settings(folder: str | Path) -> dict:
        return json.loads((Path(folder) / SETTINGS_FILE).read_text())

    def check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(f"{self.name} must be trained before it can predict or save")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.hyperparameters})"

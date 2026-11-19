"""Baseline that repeats the last known price."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from natgas.forecasting.models.AbstractModelWrapper import AbstractModelWrapper


class NaiveModel(AbstractModelWrapper):
    """Predicts the last observed price for every future date.

    For a traded price this is a hard baseline to beat over a few days, so
    every other model should be compared against it.
    """

    name = "naive"
    default_hyperparameters: dict = {}

    def __init__(self, hyperparameters: dict | None = None) -> None:
        super().__init__(hyperparameters)
        self.last_price: float | None = None

    def train(self, training_data: pd.DataFrame) -> None:
        self.last_price = float(training_data["price"].dropna().iloc[-1])
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        return pd.Series(self.last_price, index=future_features.index, name="prediction")

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        self.write_settings(folder, {"last_price": self.last_price})

    @classmethod
    def load(cls, folder: str | Path) -> NaiveModel:
        settings = cls.read_settings(folder)
        model = cls(settings["hyperparameters"])
        model.last_price = settings["last_price"]
        model.is_trained = True
        return model

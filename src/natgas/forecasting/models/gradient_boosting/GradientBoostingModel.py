"""Gradient boosted trees on recent price changes and weather."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from natgas.forecasting.models.base.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.gradient_boosting.GradientBoostingModelConfig import (
    GradientBoostingModelConfig,
)

REGRESSOR_FILE = "regressor.joblib"
HISTORY_FILE = "recent_prices.csv"


class GradientBoostingModel(AbstractModelWrapper):
    """Predicts the next daily price change, then rolls forward one day at a time.

    Features for each day are the last n_lags daily changes plus, when
    use_weather is on, that day's degree days. Trees cannot extrapolate to price
    levels they have not seen, so the model learns changes instead of levels.
    To forecast several days ahead, each predicted change becomes a lag for the
    next day.
    """

    config_class = GradientBoostingModelConfig

    def __init__(self, config: GradientBoostingModelConfig | None = None) -> None:
        super().__init__(config)
        self.regressor: HistGradientBoostingRegressor | None = None
        self.recent_prices: pd.Series | None = None

    def build_features(self, changes: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
        """Lagged changes for each date, plus that date's weather if enabled."""
        n_lags = self.config.n_lags
        lags = {f"change_lag_{k}": changes.shift(k) for k in range(1, n_lags + 1)}
        features = pd.DataFrame(lags, index=changes.index)
        for column in self.config.required_features:
            features[column] = weather.get(column, np.nan)
        return features

    def train(self, training_data: pd.DataFrame) -> None:
        n_lags = self.config.n_lags
        prices = training_data["price"].dropna()
        changes = prices.diff()
        features = self.build_features(changes, training_data.reindex(prices.index))
        rows = features.iloc[n_lags + 1 :]
        target = changes.iloc[n_lags + 1 :]
        if len(rows) < 2 * n_lags:
            raise ValueError("Not enough history to train gradient_boosting")

        self.regressor = HistGradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            max_iter=self.config.max_iter,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )
        self.regressor.fit(rows, target)
        # Keep just enough history to build lags for the first forecast day
        self.recent_prices = prices.iloc[-(n_lags + 1) :]
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        changes = list(self.recent_prices.diff().dropna())
        price = float(self.recent_prices.iloc[-1])
        predictions = []

        for day in future_features.index:
            # Most recent change first, to match change_lag_1, change_lag_2, ...
            row = {f"change_lag_{k}": changes[-k] for k in range(1, self.config.n_lags + 1)}
            for column in self.config.required_features:
                value = future_features.at[day, column] if column in future_features else np.nan
                row[column] = value
            change = float(self.regressor.predict(pd.DataFrame([row]))[0])
            price += change
            changes.append(change)
            predictions.append(price)

        return pd.Series(predictions, index=future_features.index, name="prediction")

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        folder = self.write_settings(folder)
        joblib.dump(self.regressor, folder / REGRESSOR_FILE)
        self.recent_prices.rename("price").to_csv(folder / HISTORY_FILE, index_label="date")

    @classmethod
    def load(cls, folder: str | Path) -> GradientBoostingModel:
        folder = Path(folder)
        model = cls(cls.config_from_folder(folder))
        model.regressor = joblib.load(folder / REGRESSOR_FILE)
        history = pd.read_csv(folder / HISTORY_FILE, parse_dates=["date"], index_col="date")
        model.recent_prices = history["price"]
        model.is_trained = True
        return model

"""Gradient boosted trees on recent price changes and weather."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from natgas.forecasting.models.AbstractModelWrapper import AbstractModelWrapper

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

    name = "gradient_boosting"
    default_hyperparameters: dict = {
        "n_lags": 10,
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_iter": 200,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "use_weather": True,
        "weather_columns": ["hdd", "cdd"],
        "random_state": 42,
    }

    def __init__(self, hyperparameters: dict | None = None) -> None:
        super().__init__(hyperparameters)
        self.regressor: HistGradientBoostingRegressor | None = None
        self.recent_prices: pd.Series | None = None

    @property
    def weather_columns(self) -> list[str]:
        return list(self.hyperparameters["weather_columns"]) if self.use_weather else []

    @property
    def use_weather(self) -> bool:
        return bool(self.hyperparameters["use_weather"])

    def build_features(self, changes: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
        """Lagged changes for each date, plus that date's weather if enabled."""
        lags = {f"change_lag_{k}": changes.shift(k) for k in range(1, self.n_lags + 1)}
        features = pd.DataFrame(lags, index=changes.index)
        for column in self.weather_columns:
            features[column] = weather.get(column, np.nan)
        return features

    @property
    def n_lags(self) -> int:
        return int(self.hyperparameters["n_lags"])

    def train(self, training_data: pd.DataFrame) -> None:
        prices = training_data["price"].dropna()
        changes = prices.diff()
        features = self.build_features(changes, training_data.reindex(prices.index))
        rows = features.iloc[self.n_lags + 1 :]
        target = changes.iloc[self.n_lags + 1 :]
        if len(rows) < 2 * self.n_lags:
            raise ValueError("Not enough history to train gradient_boosting")

        self.regressor = HistGradientBoostingRegressor(
            learning_rate=self.hyperparameters["learning_rate"],
            max_depth=self.hyperparameters["max_depth"],
            max_iter=self.hyperparameters["max_iter"],
            min_samples_leaf=self.hyperparameters["min_samples_leaf"],
            l2_regularization=self.hyperparameters["l2_regularization"],
            random_state=self.hyperparameters["random_state"],
        )
        self.regressor.fit(rows, target)
        # Keep just enough history to build lags for the first forecast day
        self.recent_prices = prices.iloc[-(self.n_lags + 1) :]
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        changes = list(self.recent_prices.diff().dropna())
        price = float(self.recent_prices.iloc[-1])
        predictions = []

        for day in future_features.index:
            # Most recent change first, to match change_lag_1, change_lag_2, ...
            row = {f"change_lag_{k}": changes[-k] for k in range(1, self.n_lags + 1)}
            for column in self.weather_columns:
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
        model = cls(cls.read_settings(folder)["hyperparameters"])
        model.regressor = joblib.load(folder / REGRESSOR_FILE)
        history = pd.read_csv(folder / HISTORY_FILE, parse_dates=["date"], index_col="date")
        model.recent_prices = history["price"]
        model.is_trained = True
        return model

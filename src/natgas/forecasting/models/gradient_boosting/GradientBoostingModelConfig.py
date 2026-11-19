"""Settings for the gradient boosting model."""

from __future__ import annotations

from dataclasses import dataclass, field

from natgas.forecasting.data.CalendarFeatureSource import (
    MAX_WEEKLY_HARMONICS,
    CalendarFeatureSource,
)
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper


@dataclass(frozen=True)
class GradientBoostingModelConfig(AbstractConfigWrapper):
    model_name = "gradient_boosting"

    n_lags: int = 10
    learning_rate: float = 0.05
    max_depth: int = 3
    max_iter: int = 200
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    use_weather: bool = True
    weather_columns: list[str] = field(default_factory=lambda: ["hdd", "cdd"])
    use_yearly: bool = False
    weekly_harmonics: int = 0
    use_weekend: bool = False
    random_state: int = 42

    def validate(self) -> None:
        if self.n_lags < 1:
            raise ValueError("n_lags must be at least 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.max_depth < 1 or self.max_iter < 1 or self.min_samples_leaf < 1:
            raise ValueError("max_depth, max_iter, and min_samples_leaf must be at least 1")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization cannot be negative")
        if not 0 <= self.weekly_harmonics <= MAX_WEEKLY_HARMONICS:
            raise ValueError(f"weekly_harmonics must be between 0 and {MAX_WEEKLY_HARMONICS}")

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {
            "n_lags": [5, 10],
            "learning_rate": [0.03, 0.1],
            "max_depth": [2, 4],
            "use_weather": [True, False],
            "use_yearly": [True, False],
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            "n_lags": {"low": 3, "high": 20},
            "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"low": 2, "high": 6},
            "min_samples_leaf": {"low": 10, "high": 100, "log": True},
            "use_weather": [True, False],
            "use_yearly": [True, False],
        }

    @property
    def required_features(self) -> list[str]:
        weather = list(self.weather_columns) if self.use_weather else []
        calendar = CalendarFeatureSource.column_names(
            self.use_yearly, self.weekly_harmonics, self.use_weekend
        )
        return weather + calendar

    @property
    def min_training_rows(self) -> int:
        return 3 * self.n_lags + 1

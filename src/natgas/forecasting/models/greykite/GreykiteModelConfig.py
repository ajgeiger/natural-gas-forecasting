"""Settings for the Greykite Silverkite model."""

from __future__ import annotations

from dataclasses import dataclass, field

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

FIT_ALGORITHMS = ["ridge", "linear"]


@dataclass(frozen=True)
class GreykiteModelConfig(AbstractConfigWrapper):
    """Silverkite from LinkedIn's Greykite library.

    history_years        years of history to fit on, 0 means all of it
    autoregression_lags  use the last N prices as features, 0 turns this off
    use_weather          use degree days as regressors for the forecast dates
    auto_changepoints    let the trend change direction at points Greykite finds
    fit_algorithm        ridge or linear regression for the final fit
    """

    model_name = "greykite"

    history_years: float = 3.0
    autoregression_lags: int = 3
    use_weather: bool = True
    weather_columns: list[str] = field(default_factory=lambda: ["hdd", "cdd"])
    auto_changepoints: bool = False
    fit_algorithm: str = "linear"

    def validate(self) -> None:
        if self.history_years < 0:
            raise ValueError("history_years cannot be negative")
        if self.autoregression_lags < 0:
            raise ValueError("autoregression_lags cannot be negative")
        if self.fit_algorithm not in FIT_ALGORITHMS:
            raise ValueError(f"fit_algorithm must be one of {FIT_ALGORITHMS}")

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {
            "history_years": [2.0, 4.0],
            "autoregression_lags": [0, 3],
            "use_weather": [True, False],
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            "history_years": {"low": 2.0, "high": 8.0},
            "autoregression_lags": {"low": 0, "high": 5},
            "use_weather": [True, False],
            "auto_changepoints": [True, False],
            "fit_algorithm": FIT_ALGORITHMS,
        }

    @property
    def required_features(self) -> list[str]:
        return list(self.weather_columns) if self.use_weather else []

    @property
    def min_training_rows(self) -> int:
        # About one year of trading days
        return 250

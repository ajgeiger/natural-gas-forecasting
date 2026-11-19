"""Settings for the Greykite Silverkite model."""

from __future__ import annotations

from dataclasses import dataclass

from natgas.forecasting.data.HolidayFeatureSource import MAX_WINDOW, HolidayFeatureSource
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

FIT_ALGORITHMS = ["ridge", "linear"]


@dataclass(frozen=True)
class GreykiteModelConfig(AbstractConfigWrapper):
    """Silverkite from LinkedIn's Greykite library.

    history_years        years of history to fit on, 0 means all of it
    autoregression_lags  use the last N prices as features, 0 turns this off
    use_holidays         use US holiday dummies as regressors
    holiday_window       days before and after each holiday that get a dummy, 0 to 4
    auto_changepoints    let the trend change direction at points Greykite finds
    fit_algorithm        ridge or linear regression for the final fit
    """

    model_name = "greykite"

    history_years: float = 3.0
    autoregression_lags: int = 3
    use_holidays: bool = True
    holiday_window: int = MAX_WINDOW
    auto_changepoints: bool = False
    fit_algorithm: str = "linear"

    def validate(self) -> None:
        if self.history_years < 0:
            raise ValueError("history_years cannot be negative")
        if self.autoregression_lags < 0:
            raise ValueError("autoregression_lags cannot be negative")
        if not 0 <= self.holiday_window <= MAX_WINDOW:
            raise ValueError(f"holiday_window must be between 0 and {MAX_WINDOW}")
        if self.fit_algorithm not in FIT_ALGORITHMS:
            raise ValueError(f"fit_algorithm must be one of {FIT_ALGORITHMS}")

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {
            "history_years": [2.0, 4.0],
            "autoregression_lags": [0, 3],
            "use_holidays": [True, False],
            "holiday_window": [2, 4],
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            "history_years": {"low": 2.0, "high": 8.0},
            "autoregression_lags": {"low": 0, "high": 5},
            "use_holidays": [True, False],
            "holiday_window": {"low": 0, "high": MAX_WINDOW},
            "auto_changepoints": [True, False],
            # Plain linear regression blew up on some configs, so search ridge only
            "fit_algorithm": ["ridge"],
        }

    @property
    def required_features(self) -> list[str]:
        return HolidayFeatureSource.column_names(self.holiday_window) if self.use_holidays else []

    @property
    def min_training_rows(self) -> int:
        # About one year of trading days
        return 250

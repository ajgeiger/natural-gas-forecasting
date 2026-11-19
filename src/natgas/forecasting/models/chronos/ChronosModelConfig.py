"""Settings for Amazon Chronos-2, a pretrained forecasting model."""

from __future__ import annotations

from dataclasses import dataclass, field

from natgas.forecasting.data.CalendarFeatureSource import (
    MAX_WEEKLY_HARMONICS,
    CalendarFeatureSource,
)
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

MIN_CONTEXT = 16
MAX_CONTEXT = 8192
MAX_HORIZON = 1024


@dataclass(frozen=True)
class ChronosModelConfig(AbstractConfigWrapper):
    """Chronos-2 is used zero-shot: no training, it reads recent prices and forecasts.

    model_id          Hugging Face model, a Chronos-2 checkpoint
    context_length    how many recent business days the model reads, up to 8192
    use_weather       pass heating and cooling degree days as known future covariates
    use_yearly        pass 2 yearly Fourier harmonics as known future covariates
    weekly_harmonics  weekly Fourier harmonics to pass, 0 to 2, 0 means none
    use_weekend       pass a flag for the first weekday after a weekend
    """

    model_name = "chronos"

    model_id: str = "amazon/chronos-2"
    context_length: int = 512
    use_weather: bool = True
    weather_columns: list[str] = field(default_factory=lambda: ["hdd", "cdd"])
    use_yearly: bool = False
    weekly_harmonics: int = 0
    use_weekend: bool = False
    device: str = "cpu"

    def validate(self) -> None:
        if "chronos-2" not in self.model_id:
            raise ValueError("model_id must be a Chronos-2 model, for example amazon/chronos-2")
        if not MIN_CONTEXT <= self.context_length <= MAX_CONTEXT:
            raise ValueError(f"context_length must be between {MIN_CONTEXT} and {MAX_CONTEXT}")
        if not 0 <= self.weekly_harmonics <= MAX_WEEKLY_HARMONICS:
            raise ValueError(f"weekly_harmonics must be between 0 and {MAX_WEEKLY_HARMONICS}")

    @classmethod
    def default_grid(cls) -> list[dict[str, list]]:
        base = {"context_length": [128, 512, 2048], "use_weather": [True, False]}
        # Calendar features all off, or all on with 1 or 2 weekly harmonics
        calendar = {"use_yearly": True, "weekly_harmonics": [1, 2], "use_weekend": True}
        return [base, {**base, **calendar}]

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            "context_length": {"low": 64, "high": 2048, "step": 64},
            "use_weather": [True, False],
            "use_yearly": [True, False],
            "weekly_harmonics": {"low": 0, "high": MAX_WEEKLY_HARMONICS},
            "use_weekend": [True, False],
        }

    @property
    def required_features(self) -> list[str]:
        weather = list(self.weather_columns) if self.use_weather else []
        calendar = CalendarFeatureSource.column_names(
            self.use_yearly, self.weekly_harmonics, self.use_weekend
        )
        return weather + calendar

    @property
    def max_horizon_days(self) -> int:
        return MAX_HORIZON

    @property
    def min_training_rows(self) -> int:
        return MIN_CONTEXT

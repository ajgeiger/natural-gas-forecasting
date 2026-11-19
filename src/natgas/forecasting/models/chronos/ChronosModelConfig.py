"""Settings for Amazon Chronos, a pretrained forecasting model."""

from __future__ import annotations

from dataclasses import dataclass

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

POINT_FORECASTS = ["median", "mean", "quantile"]
MODEL_IDS = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    "amazon/chronos-t5-tiny",
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
]


@dataclass(frozen=True)
class ChronosModelConfig(AbstractConfigWrapper):
    """Chronos is used zero-shot: no training, it reads recent prices and forecasts.

    model_id        Hugging Face model, bolt models are much faster on CPU
    context_length  how many recent prices the model reads
    num_samples     sample paths per forecast, only used by the t5 models
    point_forecast  median, mean, or a chosen quantile of the forecast
    quantile_level  which quantile to use when point_forecast is "quantile"
    seed            fixes the random sample paths of t5 models so runs repeat
    """

    model_name = "chronos"

    model_id: str = "amazon/chronos-bolt-small"
    context_length: int = 512
    num_samples: int = 20
    point_forecast: str = "median"
    quantile_level: float = 0.5
    device: str = "cpu"
    seed: int = 42

    def validate(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise ValueError(f"model_id must be one of {MODEL_IDS}")
        if self.context_length < 16:
            raise ValueError("context_length must be at least 16")
        if self.is_t5 and self.context_length > 512:
            raise ValueError("t5 models read at most 512 values")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.point_forecast not in POINT_FORECASTS:
            raise ValueError(f"point_forecast must be one of {POINT_FORECASTS}")
        if not 0 < self.quantile_level < 1:
            raise ValueError("quantile_level must be between 0 and 1")

    @property
    def is_t5(self) -> bool:
        return "chronos-t5" in self.model_id

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {
            "model_id": [
                "amazon/chronos-bolt-small",
                "amazon/chronos-bolt-base",
                "amazon/chronos-t5-small",
            ],
            "context_length": [128, 512],
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            "model_id": [
                "amazon/chronos-bolt-small",
                "amazon/chronos-bolt-base",
                "amazon/chronos-t5-small",
            ],
            "context_length": {"low": 64, "high": 512, "step": 64},
            "point_forecast": ["median", "mean"],
        }

    @property
    def max_horizon_days(self) -> int:
        # Chronos is trained for up to 64 steps ahead
        return 64

    @property
    def min_training_rows(self) -> int:
        return 16

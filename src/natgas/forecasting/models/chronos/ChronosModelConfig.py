"""Settings for Amazon Chronos-2, a pretrained forecasting model."""

from __future__ import annotations

from dataclasses import dataclass

from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper

MIN_CONTEXT = 16
MAX_CONTEXT = 8192
MAX_HORIZON = 1024


@dataclass(frozen=True)
class ChronosModelConfig(AbstractConfigWrapper):
    """Chronos-2 is used zero-shot: no training, it reads recent prices and forecasts.

    model_id        Hugging Face model, a Chronos-2 checkpoint
    context_length  how many recent business days the model reads, up to 8192
    """

    model_name = "chronos"

    model_id: str = "amazon/chronos-2"
    context_length: int = 512
    device: str = "cpu"

    def validate(self) -> None:
        if "chronos-2" not in self.model_id:
            raise ValueError("model_id must be a Chronos-2 model, for example amazon/chronos-2")
        if not MIN_CONTEXT <= self.context_length <= MAX_CONTEXT:
            raise ValueError(f"context_length must be between {MIN_CONTEXT} and {MAX_CONTEXT}")

    @classmethod
    def default_grid(cls) -> dict[str, list]:
        return {"context_length": [128, 512, 2048]}

    @classmethod
    def default_search_space(cls) -> dict:
        return {"context_length": {"low": 64, "high": 2048, "step": 64}}

    @property
    def max_horizon_days(self) -> int:
        return MAX_HORIZON

    @property
    def min_training_rows(self) -> int:
        return MIN_CONTEXT

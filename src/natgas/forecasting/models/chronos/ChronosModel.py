"""Amazon Chronos zero-shot forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from chronos import BaseChronosPipeline, ChronosPipeline

from natgas.forecasting.models.base.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.chronos.ChronosModelConfig import ChronosModelConfig

CONTEXT_FILE = "context.csv"


class ChronosModel(AbstractModelWrapper):
    """Pretrained time series model from Amazon, used without any fine tuning.

    train() just keeps the last context_length prices. predict() hands them to
    Chronos and returns the median, mean, or a chosen quantile for each date.
    Chronos only reads the price history, so weather is not used.

    The model weights are downloaded from Hugging Face once and cached. They
    are not saved with the model folder, only the model id is.
    """

    config_class = ChronosModelConfig
    pipelines: ClassVar[dict[str, BaseChronosPipeline]] = {}

    def __init__(self, config: ChronosModelConfig | None = None) -> None:
        super().__init__(config)
        self.context: pd.Series | None = None

    @classmethod
    def pipeline_for(cls, model_id: str, device: str) -> BaseChronosPipeline:
        """Load each Chronos model once per process and reuse it."""
        key = f"{model_id}@{device}"
        if key not in cls.pipelines:
            cls.pipelines[key] = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device, dtype=torch.float32
            )
        return cls.pipelines[key]

    def train(self, training_data: pd.DataFrame) -> None:
        self.context = training_data["price"].dropna().iloc[-self.config.context_length :]
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        pipeline = self.pipeline_for(self.config.model_id, self.config.device)
        context = torch.tensor(self.context.to_numpy(), dtype=torch.float32)
        options = {}
        if isinstance(pipeline, ChronosPipeline):
            options["num_samples"] = self.config.num_samples

        # t5 models draw random sample paths, a fixed seed makes backtests repeatable
        torch.manual_seed(self.config.seed)
        level = 0.5 if self.config.point_forecast == "median" else self.config.quantile_level
        with torch.no_grad():
            quantiles, mean = pipeline.predict_quantiles(
                context,
                prediction_length=len(future_features),
                quantile_levels=[level],
                **options,
            )
        values = mean[0] if self.config.point_forecast == "mean" else quantiles[0, :, 0]
        return pd.Series(values.numpy(), index=future_features.index, name="prediction")

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        folder = self.write_settings(folder)
        self.context.rename("price").to_csv(folder / CONTEXT_FILE, index_label="date")

    @classmethod
    def load(cls, folder: str | Path) -> ChronosModel:
        folder = Path(folder)
        model = cls(cls.config_from_folder(folder))
        context = pd.read_csv(folder / CONTEXT_FILE, parse_dates=["date"], index_col="date")
        model.context = context["price"]
        model.is_trained = True
        return model

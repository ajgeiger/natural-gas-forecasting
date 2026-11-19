"""Amazon Chronos-2 zero-shot forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from chronos import Chronos2Pipeline

from natgas.forecasting.models.base.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.chronos.ChronosModelConfig import ChronosModelConfig

CONTEXT_FILE = "context.csv"
SERIES_ID = "ng"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "target"
ID_COLUMN = "id"


class ChronosModel(AbstractModelWrapper):
    """Pretrained Chronos-2 model from Amazon, used without any fine tuning.

    train() keeps the last context_length prices on a business day calendar.
    predict() hands them to Chronos-2 and returns the median for each date.

    The model weights are downloaded from Hugging Face once and cached. They
    are not saved with the model folder, only the model id is.
    """

    config_class = ChronosModelConfig
    pipelines: ClassVar[dict[str, Chronos2Pipeline]] = {}

    def __init__(self, config: ChronosModelConfig | None = None) -> None:
        super().__init__(config)
        self.context: pd.Series | None = None

    @classmethod
    def pipeline_for(cls, model_id: str, device: str) -> Chronos2Pipeline:
        """Load Chronos-2 once per process and reuse it."""
        key = f"{model_id}@{device}"
        if key not in cls.pipelines:
            # Prefer the local cache so parallel backtest workers do not
            # contact Hugging Face on every process start.
            options = {"device_map": device, "dtype": torch.float32}
            try:
                cls.pipelines[key] = Chronos2Pipeline.from_pretrained(
                    model_id, local_files_only=True, **options
                )
            except OSError:
                cls.pipelines[key] = Chronos2Pipeline.from_pretrained(model_id, **options)
        return cls.pipelines[key]

    def train(self, training_data: pd.DataFrame) -> None:
        prices = training_data["price"].dropna()
        # Chronos-2 needs a regular time grid, so holidays carry the previous
        # price forward, the same way Greykite does.
        calendar = pd.bdate_range(prices.index[0], prices.index[-1])
        self.context = prices.reindex(calendar).ffill().iloc[-self.config.context_length :]
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        pipeline = self.pipeline_for(self.config.model_id, self.config.device)
        future_calendar = pd.bdate_range(
            self.context.index[-1] + pd.Timedelta(days=1), future_features.index.max()
        )
        forecast = pipeline.predict_df(
            self.context_frame(),
            prediction_length=len(future_calendar),
            quantile_levels=[0.5],
            id_column=ID_COLUMN,
            timestamp_column=TIME_COLUMN,
            target=TARGET_COLUMN,
            context_length=self.config.context_length,
            freq="B",
        )
        # The median is the point forecast. Chronos-2 also returns it as "predictions".
        values = forecast["0.5"].to_numpy()
        dates = pd.DatetimeIndex(forecast[TIME_COLUMN])
        return pd.Series(values, index=dates, name="prediction").reindex(future_features.index)

    def context_frame(self) -> pd.DataFrame:
        frame = self.context.rename(TARGET_COLUMN).rename_axis(TIME_COLUMN).reset_index()
        frame.insert(0, ID_COLUMN, SERIES_ID)
        return frame

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        folder = self.write_settings(folder)
        self.context.to_csv(folder / CONTEXT_FILE, index_label="date")

    @classmethod
    def load(cls, folder: str | Path) -> ChronosModel:
        folder = Path(folder)
        model = cls(cls.config_from_folder(folder))
        context = pd.read_csv(folder / CONTEXT_FILE, parse_dates=["date"], index_col="date")
        model.context = context["price"]
        model.is_trained = True
        return model

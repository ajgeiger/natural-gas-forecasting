"""Amazon Chronos-2 zero-shot forecasts, with optional weather covariates."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from chronos import Chronos2Pipeline

from models.chronos.ChronosModelConfig import ChronosModelConfig
from models.config.AbstractModelWrapper import AbstractModelWrapper

CONTEXT_FILE = "context.csv"
SERIES_ID = "ng"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "target"
ID_COLUMN = "id"
# The median is the point forecast. The 10th and 90th percentiles make an 80 percent interval.
LOWER_QUANTILE = 0.1
MEDIAN_QUANTILE = 0.5
UPPER_QUANTILE = 0.9


class ChronosModel(AbstractModelWrapper):
    """Pretrained Chronos-2 model from Amazon, used without any fine tuning.

    train() keeps the last context_length prices, plus any weather and
    calendar columns the config turns on, on a business day calendar.
    predict() hands them to Chronos-2, with those columns for the forecast
    dates passed as known future covariates. The median is the point
    forecast, and predict_with_interval() also returns the 10th and 90th
    percentiles.

    The model weights are downloaded from Hugging Face once and cached. They
    are not saved with the model folder, only the model id is.
    """

    config_class = ChronosModelConfig
    makes_intervals = True
    pipelines: ClassVar[dict[str, Chronos2Pipeline]] = {}

    def __init__(self, config: ChronosModelConfig | None = None) -> None:
        super().__init__(config)
        self.context: pd.DataFrame | None = None

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
        columns = ["price", *self.config.required_features]
        history = training_data[columns].dropna(subset=["price"])
        # Chronos-2 needs a regular time grid, so holidays carry the previous
        # row forward, the same way Greykite does.
        calendar = pd.bdate_range(history.index[0], history.index[-1])
        history = history.reindex(calendar).ffill()
        for column in self.config.required_features:
            history[column] = history[column].bfill()
        self.context = history.iloc[-self.config.context_length :]
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        return self.predict_with_interval(future_features)["prediction"]

    def predict_with_interval(self, future_features: pd.DataFrame) -> pd.DataFrame:
        self.check_trained()
        pipeline = self.pipeline_for(self.config.model_id, self.config.device)
        future_calendar = pd.bdate_range(
            self.context.index[-1] + pd.Timedelta(days=1), future_features.index.max()
        )
        forecast = pipeline.predict_df(
            self.context_frame(),
            future_df=self.future_covariates(future_features, future_calendar),
            prediction_length=len(future_calendar),
            quantile_levels=[LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE],
            id_column=ID_COLUMN,
            timestamp_column=TIME_COLUMN,
            target=TARGET_COLUMN,
            context_length=self.config.context_length,
            freq="B",
        )
        frame = pd.DataFrame(
            {
                "prediction": forecast[str(MEDIAN_QUANTILE)].to_numpy(),
                "lower": forecast[str(LOWER_QUANTILE)].to_numpy(),
                "upper": forecast[str(UPPER_QUANTILE)].to_numpy(),
            },
            index=pd.DatetimeIndex(forecast[TIME_COLUMN]),
        )
        return frame.reindex(future_features.index)

    def context_frame(self) -> pd.DataFrame:
        frame = self.context.rename_axis(TIME_COLUMN).reset_index()
        frame = frame.rename(columns={"price": TARGET_COLUMN})
        frame.insert(0, ID_COLUMN, SERIES_ID)
        return frame

    def future_covariates(
        self, future_features: pd.DataFrame, future_calendar: pd.DatetimeIndex
    ) -> pd.DataFrame | None:
        if not self.config.required_features:
            return None
        future = pd.DataFrame(index=future_calendar)
        for column in self.config.required_features:
            values = future_features[column].reindex(future_calendar).ffill().bfill()
            future[column] = values.fillna(self.context[column].iloc[-1])
        future = future.rename_axis(TIME_COLUMN).reset_index()
        future.insert(0, ID_COLUMN, SERIES_ID)
        return future

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        folder = self.write_settings(folder)
        self.context.to_csv(folder / CONTEXT_FILE, index_label="date")

    @classmethod
    def load(cls, folder: str | Path) -> ChronosModel:
        folder = Path(folder)
        model = cls(cls.config_from_folder(folder))
        model.context = pd.read_csv(folder / CONTEXT_FILE, parse_dates=["date"], index_col="date")
        model.is_trained = True
        return model

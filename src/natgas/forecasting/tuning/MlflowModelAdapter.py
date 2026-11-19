"""Lets MLflow load and run any AbstractModelWrapper."""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd

from natgas.forecasting.models.ModelCatalog import ModelCatalog

MODEL_FOLDER_KEY = "model_folder"


class MlflowModelAdapter(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around a saved model folder.

    Input: a DataFrame with a "date" column and any feature columns the model
    uses (for example hdd and cdd from the weather forecast).
    Output: a DataFrame with date and prediction.
    """

    def load_context(self, context) -> None:
        self.model = ModelCatalog.load(context.artifacts[MODEL_FOLDER_KEY])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        features = model_input.copy()
        features["date"] = pd.to_datetime(features["date"])
        features = features.set_index("date").sort_index()
        prediction = self.model.predict(features)
        return prediction.rename_axis("date").reset_index(name="prediction")

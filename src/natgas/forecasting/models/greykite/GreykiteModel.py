"""LinkedIn Greykite Silverkite forecasts, with holiday dummies as regressors."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
from greykite.framework.templates.autogen.forecast_config import (
    ComputationParam,
    EvaluationPeriodParam,
    ForecastConfig,
    MetadataParam,
    ModelComponentsParam,
)
from greykite.framework.templates.forecaster import Forecaster

from natgas.forecasting.models.base.AbstractModelWrapper import AbstractModelWrapper
from natgas.forecasting.models.greykite.GreykiteModelConfig import GreykiteModelConfig

HISTORY_FILE = "history.csv"
TIME_COLUMN = "ts"
VALUE_COLUMN = "price"
NO_EVENTS = {"holiday_lookup_countries": [], "holidays_to_model_separately": []}


class GreykiteModel(AbstractModelWrapper):
    """Silverkite: trend, seasonality, recent prices, and holiday regressors.

    Greykite needs the regressor values for the forecast dates at fit time, so
    train() prepares the history and predict() fits and forecasts in one step.
    Only data on or before the as-of date is used for the target either way.

    Greykite expects an even time grid, so the history is put on a business
    day calendar and exchange holidays carry the previous price forward.
    Holiday dummies that are never 1 in the history are left out of the fit.
    When holiday dummies are used, Greykite's own holiday events are turned
    off so holidays are not counted twice.
    """

    config_class = GreykiteModelConfig

    def __init__(self, config: GreykiteModelConfig | None = None) -> None:
        super().__init__(config)
        self.history: pd.DataFrame | None = None

    def train(self, training_data: pd.DataFrame) -> None:
        columns = [VALUE_COLUMN, *self.config.required_features]
        history = training_data[columns].dropna(subset=[VALUE_COLUMN])
        calendar = pd.bdate_range(history.index[0], history.index[-1])
        history = history.reindex(calendar)
        history[VALUE_COLUMN] = history[VALUE_COLUMN].ffill()
        history = history.fillna(0.0)
        if self.config.history_years > 0:
            start = calendar[-1] - pd.DateOffset(days=round(365.25 * self.config.history_years))
            history = history.loc[start:]
        self.history = history
        self.is_trained = True

    def predict(self, future_features: pd.DataFrame) -> pd.Series:
        self.check_trained()
        future_calendar = pd.bdate_range(
            self.history.index[-1] + pd.Timedelta(days=1), future_features.index.max()
        )
        regressors = self.regressor_columns()
        future = future_features[regressors].reindex(future_calendar, fill_value=0.0)

        frame = (
            pd.concat([self.history[[VALUE_COLUMN, *regressors]], future])
            .rename_axis(TIME_COLUMN)
            .reset_index()
        )
        forecast = self.run_greykite(frame, regressors, len(future_calendar))
        prediction = forecast.set_index(TIME_COLUMN)["forecast"].reindex(future_features.index)
        return prediction.rename("prediction")

    def regressor_columns(self) -> list[str]:
        """Holiday dummies that are 1 at least once in the history."""
        return [column for column in self.config.required_features if self.history[column].any()]

    def run_greykite(
        self, frame: pd.DataFrame, regressors: list[str], horizon: int
    ) -> pd.DataFrame:
        """Fit Silverkite on the history rows and forecast the future rows."""
        components = ModelComponentsParam(
            regressors={"regressor_cols": regressors} if regressors else None,
            events=NO_EVENTS if self.config.use_holidays else None,
            autoregression=self.autoregression_settings(),
            changepoints=(
                {"changepoints_dict": {"method": "auto"}} if self.config.auto_changepoints else None
            ),
            custom={"fit_algorithm_dict": {"fit_algorithm": self.config.fit_algorithm}},
        )
        forecast_config = ForecastConfig(
            model_template="SILVERKITE",
            forecast_horizon=horizon,
            coverage=None,
            metadata_param=MetadataParam(time_col=TIME_COLUMN, value_col=VALUE_COLUMN, freq="B"),
            model_components_param=components,
            evaluation_period_param=EvaluationPeriodParam(test_horizon=0, cv_max_splits=0),
            computation_param=ComputationParam(n_jobs=1, verbose=0),
        )
        # Greykite is chatty, keep backtest logs readable
        logging.getLogger("greykite").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Forecaster().run_forecast_config(df=frame, config=forecast_config)
        forecast = result.forecast.df
        return forecast[forecast[TIME_COLUMN] > self.history.index[-1]]

    def autoregression_settings(self) -> dict | None:
        """Lags 1 to N of the price. Lags shorter than the horizon are simulated forward."""
        lags = self.config.autoregression_lags
        if lags == 0:
            return None
        orders = list(range(1, lags + 1))
        return {"autoreg_dict": {"lag_dict": {"orders": orders}}, "fast_simulation": True}

    def save(self, folder: str | Path) -> None:
        self.check_trained()
        folder = self.write_settings(folder)
        self.history.to_csv(folder / HISTORY_FILE, index_label="date")

    @classmethod
    def load(cls, folder: str | Path) -> GreykiteModel:
        folder = Path(folder)
        model = cls(cls.config_from_folder(folder))
        model.history = pd.read_csv(folder / HISTORY_FILE, parse_dates=["date"], index_col="date")
        model.history.index.freq = None
        model.is_trained = True
        return model

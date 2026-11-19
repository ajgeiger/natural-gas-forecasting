"""Accuracy scores for a set of forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForecastMetrics:
    """Scores predictions against actuals.

    mae    average absolute error, in dollars per MMBtu
    rmse   like mae but punishes large misses more
    mape   average absolute error as a percent of the actual price
    wape   total absolute error as a percent of total actual price, steadier than mape
    bias   average of predicted minus actual, positive means forecasts run high
    """

    names = ["mae", "rmse", "mape", "wape", "bias"]

    @staticmethod
    def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        if actual.shape != predicted.shape or actual.size == 0:
            raise ValueError("actual and predicted must be the same non-empty length")
        error = predicted - actual
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mape": float(np.mean(np.abs(error) / np.abs(actual)) * 100),
            "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100),
            "bias": float(np.mean(error)),
        }

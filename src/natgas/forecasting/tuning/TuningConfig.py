"""Settings for a tuning run, read from config/tuning.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

LOWER_IS_BETTER = ["mae", "rmse", "mape", "wape"]


@dataclass(frozen=True)
class TuningConfig:
    """Which models and hyperparameters to try, how to backtest, and where to log."""

    model_grids: dict[str, dict[str, list]]
    horizon_days: int = 5
    backtest_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2023-01-01"))
    backtest_end: pd.Timestamp | None = None
    step_days: int = 30
    selection_metric: str = "mae"
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "natgas-backtests"
    registered_model_name: str = "natgas-price-forecaster"
    model_alias: str = "champion"
    target_column: str = "high"

    def __post_init__(self) -> None:
        if self.selection_metric not in LOWER_IS_BETTER:
            raise ValueError(f"selection_metric must be one of {LOWER_IS_BETTER}")
        if not self.model_grids:
            raise ValueError("tuning config needs at least one model")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TuningConfig:
        raw = yaml.safe_load(Path(path).read_text())
        backtest = raw.get("backtest", {})
        mlflow_settings = raw.get("mlflow", {})
        end = backtest.get("end")
        return cls(
            model_grids={name: grid or {} for name, grid in raw["models"].items()},
            horizon_days=int(backtest.get("horizon_days", 5)),
            backtest_start=pd.Timestamp(str(backtest.get("start", "2023-01-01"))),
            backtest_end=pd.Timestamp(str(end)) if end else None,
            step_days=int(backtest.get("step_days", 30)),
            selection_metric=raw.get("selection_metric", "mae"),
            tracking_uri=mlflow_settings.get("tracking_uri", "sqlite:///mlflow.db"),
            experiment_name=mlflow_settings.get("experiment_name", "natgas-backtests"),
            registered_model_name=mlflow_settings.get(
                "registered_model_name", "natgas-price-forecaster"
            ),
            model_alias=mlflow_settings.get("model_alias", "champion"),
            target_column=raw.get("target_column", "high"),
        )

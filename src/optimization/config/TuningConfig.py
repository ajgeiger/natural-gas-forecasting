"""Settings for a tuning run, read from config/tuning.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

LOWER_IS_BETTER = ["mae", "rmse", "mape", "wape"]
OPTIMIZERS = ["grid", "bayesian"]


@dataclass(frozen=True)
class TuningConfig:
    """Which models to search, how to search, how to backtest, and where to log.

    model_settings maps a model name to its optional "grid" (for grid search)
    and "search_space" (for Bayesian search). A model listed with no settings
    uses its config class defaults.
    """

    model_settings: dict[str, dict]
    optimizer: str = "grid"
    n_trials: int = 10
    seed: int = 42
    horizon_days: int = 5
    backtest_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2022-01-01"))
    backtest_end: pd.Timestamp | None = None
    step_days: int = 14
    n_jobs: int = 1
    selection_metric: str = "mae"
    min_improvement: float = 0.01
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "natgas-backtests"
    registered_model_name: str = "natgas-price-forecaster"
    model_alias: str = "champion"
    target_column: str = "high"

    def __post_init__(self) -> None:
        if self.selection_metric not in LOWER_IS_BETTER:
            raise ValueError(f"selection_metric must be one of {LOWER_IS_BETTER}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if not self.model_settings:
            raise ValueError("tuning config needs at least one model")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TuningConfig:
        raw = yaml.safe_load(Path(path).read_text())
        backtest = raw.get("backtest", {})
        search = raw.get("search", {})
        publish = raw.get("publish", {})
        mlflow_settings = raw.get("mlflow", {})
        end = backtest.get("end")
        return cls(
            model_settings={name: settings or {} for name, settings in raw["models"].items()},
            optimizer=search.get("optimizer", "grid"),
            n_trials=int(search.get("n_trials", 10)),
            seed=int(search.get("seed", 42)),
            selection_metric=search.get("selection_metric", "mae"),
            horizon_days=int(backtest.get("horizon_days", 5)),
            backtest_start=pd.Timestamp(str(backtest.get("start", "2022-01-01"))),
            backtest_end=pd.Timestamp(str(end)) if end else None,
            step_days=int(backtest.get("step_days", 14)),
            n_jobs=int(backtest.get("n_jobs", 1)),
            min_improvement=float(publish.get("min_improvement", 0.01)),
            model_alias=publish.get("alias", "champion"),
            registered_model_name=publish.get("registered_model_name", "natgas-price-forecaster"),
            tracking_uri=mlflow_settings.get("tracking_uri", "sqlite:///mlflow.db"),
            experiment_name=mlflow_settings.get("experiment_name", "natgas-backtests"),
            target_column=raw.get("target_column", "high"),
        )

    def with_values(self, **changes) -> TuningConfig:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return TuningConfig(**{**values, **changes})

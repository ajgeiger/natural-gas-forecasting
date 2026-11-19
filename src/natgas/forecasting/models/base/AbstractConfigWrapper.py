"""Shared contract for every model's settings."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, ClassVar


@dataclass(frozen=True)
class AbstractConfigWrapper(ABC):
    """Hyperparameters for one model, plus what the backtester needs to know.

    Each model has a matching config class. The config:
      - holds the hyperparameters, with defaults, and checks them in validate()
      - gives the default grid for grid search and the default search space for
        Bayesian search
      - converts to and from plain dicts for YAML files and MLflow params
      - tells the backtester which feature columns it needs, the longest
        horizon it supports, and how much history it needs to train
    """

    model_name: ClassVar[str] = ""

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Raise ValueError if any hyperparameter is out of range."""

    @classmethod
    @abstractmethod
    def default_grid(cls) -> dict[str, list]:
        """Options for grid search: every combination is tried."""

    @classmethod
    @abstractmethod
    def default_search_space(cls) -> dict[str, Any]:
        """Options for Bayesian search.

        A list means pick one of these values. A dict with low and high means a
        number range, for example {"low": 0.01, "high": 0.3, "log": True}.
        """

    @property
    def required_features(self) -> list[str]:
        """Feature columns the data provider must supply."""
        return []

    @property
    def max_horizon_days(self) -> int | None:
        """Longest forecast the model supports. None means no limit."""
        return None

    @property
    def min_training_rows(self) -> int:
        """Fewest training rows the model can work with."""
        return 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mlflow_params(self) -> dict[str, Any]:
        """Flat params for MLflow. Lists become comma separated text."""
        params = {}
        for name, value in self.to_dict().items():
            params[name] = ",".join(map(str, value)) if isinstance(value, list) else value
        return params

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None) -> AbstractConfigWrapper:
        """Build a config from a dict. Missing names keep their defaults."""
        values = values or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"{cls.model_name} does not accept: {sorted(unknown)}")
        return cls(**values)

    def with_values(self, **changes: Any) -> AbstractConfigWrapper:
        """Copy of this config with some values changed."""
        return replace(self, **changes)

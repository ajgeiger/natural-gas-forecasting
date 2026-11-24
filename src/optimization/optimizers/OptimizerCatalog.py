"""Builds an optimizer from its name in tuning.yaml."""

from __future__ import annotations

from backtesting.Backtester import Backtester
from optimization.optimizers.AbstractOptimizer import AbstractOptimizer
from optimization.optimizers.BayesianOptimizer import BayesianOptimizer
from optimization.optimizers.GridSearchOptimizer import GridSearchOptimizer


class OptimizerCatalog:
    """Maps "grid" and "bayesian" to their optimizer classes.

    Add a new search method here and it can be picked in tuning.yaml.
    """

    optimizers: dict[str, type[AbstractOptimizer]] = {
        GridSearchOptimizer.method_name: GridSearchOptimizer,
        BayesianOptimizer.method_name: BayesianOptimizer,
    }

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.optimizers)

    @classmethod
    def build(
        cls,
        name: str,
        backtester: Backtester,
        selection_metric: str = "mae",
        n_trials: int = 10,
        seed: int = 42,
    ) -> AbstractOptimizer:
        if name not in cls.optimizers:
            raise KeyError(f"Unknown optimizer {name!r}. Choose from {cls.names()}")
        if name == BayesianOptimizer.method_name:
            return BayesianOptimizer(backtester, selection_metric, n_trials, seed)
        return cls.optimizers[name](backtester, selection_metric)

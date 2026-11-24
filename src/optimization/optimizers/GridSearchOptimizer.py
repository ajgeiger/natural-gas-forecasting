"""Tries every combination in a grid."""

from __future__ import annotations

from models.config.AbstractConfigWrapper import AbstractConfigWrapper
from optimization.optimizers.AbstractOptimizer import (
    AbstractOptimizer,
)
from optimization.search_space.HyperparameterGrid import HyperparameterGrid


class GridSearchOptimizer(AbstractOptimizer):
    """Backtests every combination of the grid. Simple and complete, but the
    number of runs grows quickly with each new option."""

    method_name = "grid"
    space_key = "grid"

    def default_space(self, config_class: type[AbstractConfigWrapper]) -> dict:
        return config_class.default_grid()

    def search(self, config_class: type[AbstractConfigWrapper], space: dict) -> None:
        for values in HyperparameterGrid(space).configs():
            self.evaluate(config_class.from_dict(values))

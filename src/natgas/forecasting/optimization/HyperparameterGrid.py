"""Turns a grid of options into every combination to try."""

from __future__ import annotations

from itertools import product


class HyperparameterGrid:
    """Expands {"max_depth": [3, 5], "n_lags": [5]} into a list of configs.

    A value that is not a list is treated as a single fixed choice. A list of
    grids is expanded one grid at a time and the configs are joined, which
    keeps related options together instead of crossing every one of them.
    """

    def __init__(self, grid: dict | list[dict] | None = None) -> None:
        self.grids = grid if isinstance(grid, list) else [grid or {}]

    def configs(self) -> list[dict]:
        configs = []
        for grid in self.grids:
            names = list(grid)
            choices = [value if isinstance(value, list) else [value] for value in grid.values()]
            configs += [dict(zip(names, combo, strict=True)) for combo in product(*choices)]
        return configs

    def __len__(self) -> int:
        return len(self.configs())

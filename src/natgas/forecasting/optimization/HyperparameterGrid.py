"""Turns a grid of options into every combination to try."""

from __future__ import annotations

from itertools import product


class HyperparameterGrid:
    """Expands {"max_depth": [3, 5], "n_lags": [5]} into a list of configs.

    A value that is not a list is treated as a single fixed choice.
    """

    def __init__(self, grid: dict | None = None) -> None:
        self.grid = grid or {}

    def configs(self) -> list[dict]:
        names = list(self.grid)
        choices = [value if isinstance(value, list) else [value] for value in self.grid.values()]
        return [dict(zip(names, combo, strict=True)) for combo in product(*choices)]

    def __len__(self) -> int:
        return len(self.configs())

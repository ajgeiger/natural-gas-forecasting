"""Bayesian search with Optuna: each trial learns from the ones before it."""

from __future__ import annotations

import math

import optuna

from backtesting.Backtester import Backtester
from models.config.AbstractConfigWrapper import AbstractConfigWrapper
from optimization.optimizers.AbstractOptimizer import (
    AbstractOptimizer,
)


class BayesianOptimizer(AbstractOptimizer):
    """Uses Optuna's TPE sampler to pick promising configs.

    A list in the space is a set of choices. A dict with low and high is a
    number range, with optional step and log. Configs that come up twice are
    scored once, and a space with few combinations gets fewer trials.
    """

    method_name = "bayesian"
    space_key = "search_space"

    def __init__(
        self,
        backtester: Backtester,
        selection_metric: str = "mae",
        n_trials: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(backtester, selection_metric)
        self.n_trials = n_trials
        self.seed = seed
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def default_space(self, config_class: type[AbstractConfigWrapper]) -> dict:
        return config_class.default_search_space()

    @staticmethod
    def suggest(trial: optuna.Trial, name: str, option) -> object:
        """Ask Optuna for one value."""
        if isinstance(option, list):
            return trial.suggest_categorical(name, option)
        if not isinstance(option, dict):
            return option
        low, high = option["low"], option["high"]
        log = bool(option.get("log", False))
        step = option.get("step")
        if isinstance(low, int) and isinstance(high, int) and not isinstance(step, float):
            return trial.suggest_int(name, low, high, step=1 if log else (step or 1), log=log)
        return trial.suggest_float(name, float(low), float(high), step=step, log=log)

    def trial_count(self, space: dict) -> int:
        """No more trials than there are distinct combinations."""
        if any(isinstance(option, dict) for option in space.values()):
            return self.n_trials
        combinations = math.prod(
            len(option) for option in space.values() if isinstance(option, list)
        )
        return min(self.n_trials, combinations)

    def search(self, config_class: type[AbstractConfigWrapper], space: dict) -> None:
        def objective(trial: optuna.Trial) -> float:
            values = {name: self.suggest(trial, name, option) for name, option in space.items()}
            return self.evaluate(config_class.from_dict(values))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=self.trial_count(space))

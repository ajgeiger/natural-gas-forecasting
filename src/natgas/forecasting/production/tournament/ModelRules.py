"""Domain rules every model must follow before it is scored or can win an event."""

from __future__ import annotations

from natgas.forecasting.data.WeatherFeatureSource import COLUMNS as WEATHER_COLUMNS
from natgas.forecasting.models.base.AbstractConfigWrapper import AbstractConfigWrapper


class ModelRules:
    """Chronos-2 must use weather covariates, and Greykite must not use weather.

    Chronos-2 reads degree days as known future covariates, which is the point
    of using it here. Greykite drifts with weather, so it uses holiday dummies
    instead.
    """

    @staticmethod
    def uses_weather(config: AbstractConfigWrapper) -> bool:
        return bool(set(config.required_features) & set(WEATHER_COLUMNS))

    @staticmethod
    def problem(config: AbstractConfigWrapper) -> str | None:
        """The broken rule, or None when the config follows every rule."""
        uses_weather = ModelRules.uses_weather(config)
        if config.model_name == "chronos" and not uses_weather:
            return "Chronos-2 must use weather covariates"
        if config.model_name == "greykite" and uses_weather:
            return "Greykite must not use weather"
        return None

    @staticmethod
    def passes(config: AbstractConfigWrapper) -> bool:
        return ModelRules.problem(config) is None

    @staticmethod
    def check(config: AbstractConfigWrapper) -> None:
        problem = ModelRules.problem(config)
        if problem:
            raise ValueError(problem)

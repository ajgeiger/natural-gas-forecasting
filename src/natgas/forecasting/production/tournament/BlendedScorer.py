"""Scores contenders on recent production forecasts blended with their backtests."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from natgas.db import ForecastActual, session_scope
from natgas.forecasting.production.tournament.BlendedScore import BlendedScore
from natgas.forecasting.production.tournament.Contender import (
    BASELINE,
    CHALLENGER,
    CHAMPION,
    Contender,
)
from natgas.forecasting.production.tournament.TournamentConfig import ALL_DAYS


class BlendedScorer:
    """The rolling scorecard used by both the trials and the events.

    A model is judged on the last 30 as-of dates whose forecasts have matured
    in that bucket. A new model has little production history, so its
    backtest fills the rest of the window:

        prod_weight = min(prod_days, 30) / 30
        blended_mae = prod_weight * prod_mae + (1 - prod_weight) * backtest_mae

    On day one the score is the backtest. After 30 matured days it is pure
    production. Forecasts count by family, model, and version, under either
    family stage, so a challenger brings its record with it when promoted.
    """

    def __init__(self, db_url: str, window_days: int = 30) -> None:
        self.db_url = db_url
        self.window_days = window_days

    @staticmethod
    def blend(
        prod_mae: float, prod_days: int, backtest_mae: float, window_days: int = 30
    ) -> tuple[float, float]:
        """prod_weight and blended_mae."""
        prod_weight = min(prod_days, window_days) / float(window_days)
        prod_mae = prod_mae if prod_days > 0 else 0.0
        return prod_weight, prod_weight * prod_mae + (1 - prod_weight) * backtest_mae

    @staticmethod
    def stages_for(contender: Contender) -> list[str]:
        """Stages whose forecasts count toward this contender's record."""
        if contender.role == BASELINE:
            return [contender.stage]
        return [f"{contender.family}:{role}" for role in (CHAMPION, CHALLENGER)]

    def errors(self, contender: Contender, bucket: str, as_of_date) -> pd.DataFrame:
        """as_of_date and error of every matured forecast of this version in the bucket."""
        query = select(ForecastActual.as_of_date, ForecastActual.error).where(
            ForecastActual.stage.in_(self.stages_for(contender)),
            ForecastActual.model == contender.model,
            ForecastActual.model_version == contender.version,
            ForecastActual.as_of_date <= as_of_date,
        )
        if bucket != ALL_DAYS:
            query = query.where(ForecastActual.horizon_bucket == bucket)
        with session_scope(self.db_url) as session:
            rows = session.execute(query).all()
        return pd.DataFrame(rows, columns=["as_of_date", "error"])

    def score(
        self,
        contender: Contender,
        bucket: str,
        as_of_date,
        fallback_mae: dict[str, float] | None = None,
    ) -> BlendedScore:
        """Blended score for one bucket. fallback_mae is the baseline backtest MAE."""
        errors = self.errors(contender, bucket, as_of_date)
        recent = sorted(errors["as_of_date"].unique())[-self.window_days :]
        window = errors[errors["as_of_date"].isin(recent)]
        prod_days = len(recent)
        prod_mae = float(window["error"].abs().mean()) if prod_days else 0.0

        backtest_source = "model"
        backtest_mae = contender.backtest_mae.get(bucket)
        if backtest_mae is None and fallback_mae and bucket in fallback_mae:
            backtest_mae = fallback_mae[bucket]
            backtest_source = "naive"
        if backtest_mae is None:
            raise ValueError(f"{contender.label} has no backtest MAE for {bucket}")

        prod_weight, blended = self.blend(prod_mae, prod_days, backtest_mae, self.window_days)
        return BlendedScore(
            stage=contender.stage,
            model=contender.model,
            model_version=str(contender.version),
            horizon_bucket=bucket,
            prod_days=prod_days,
            prod_mae=prod_mae,
            backtest_mae=float(backtest_mae),
            backtest_source=backtest_source,
            prod_weight=prod_weight,
            blended_mae=float(blended),
        )

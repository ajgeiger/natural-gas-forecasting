"""Tests for production scoring, the backtest export, actuals, and monitoring. No network."""

from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import text

from natgas.config import Settings
from natgas.db import Price, WeatherDaily, get_engine, session_scope, upsert
from natgas.forecasting.backtesting import Backtester
from natgas.forecasting.data import (
    ForecastDataProvider,
    ForecastWeatherFeatureSource,
    TradingCalendar,
    WeatherFeatureSource,
)
from natgas.forecasting.models import (
    ChronosModelConfig,
    GradientBoostingModelConfig,
    GreykiteModelConfig,
    NaiveModelConfig,
)
from natgas.forecasting.optimization import GridSearchOptimizer
from natgas.forecasting.production import (
    ActualsRefresher,
    BacktestExporter,
    ChallengerPublisher,
    ForecastMonitor,
    GridSearchRuns,
    ProductionScorer,
    ScoringPipeline,
)
from natgas.forecasting.tuning import ModelPublisher, PublishDecision, TuningConfig

MODEL_NAME = "test-forecaster"
LAST_PRICE = date(2024, 6, 28)


def write_prices(settings: Settings, dates: pd.DatetimeIndex, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    highs = 3 + np.cumsum(rng.normal(0, 0.05, len(dates)))
    rows = [
        {"date": day.date(), "high": float(high), "source": "test"}
        for day, high in zip(dates, highs, strict=True)
    ]
    with session_scope(settings.db_url) as session:
        upsert(session, Price, rows)


def read(settings: Settings, query: str) -> list:
    with get_engine(settings.db_url).connect() as connection:
        return connection.execute(text(query)).all()


@pytest.fixture
def filled_settings(settings: Settings) -> Settings:
    """A database with two years of prices and weather, plus 14 days of forecast weather."""
    prices = pd.bdate_range(end=pd.Timestamp(LAST_PRICE), periods=500)
    write_prices(settings, prices)
    weather_days = pd.date_range(prices[0], pd.Timestamp(LAST_PRICE) + pd.Timedelta(days=14))
    hdd = 15 + 15 * np.cos(2 * np.pi * weather_days.dayofyear / 365.25)
    rows = [
        {
            "date": day.date(),
            "location": "chicago",
            "tavg_f": 65 - value,
            "hdd": float(value),
            "cdd": 0.0,
            "is_forecast": day.date() > LAST_PRICE,
        }
        for day, value in zip(weather_days, hdd, strict=True)
    ]
    with session_scope(settings.db_url) as session:
        upsert(session, WeatherDaily, rows)
    return settings


@pytest.fixture
def tuning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TuningConfig:
    # MLflow writes artifacts to ./mlruns, so keep them in the temp folder
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    # Earlier tests leave their own experiment active
    mlflow.set_experiment("test-production")
    return TuningConfig(
        model_settings={"naive": {}},
        horizon_days=30,
        tracking_uri=tracking_uri,
        registered_model_name=MODEL_NAME,
    )


def publish(provider: ForecastDataProvider, config, alias: str) -> str:
    dates = provider.backtest_dates("2024-01-01", "2024-04-30", step_days=60)
    report = Backtester(provider, dates).run(config)
    with mlflow.start_run():
        publisher = ModelPublisher(provider, MODEL_NAME, alias)
        return publisher.publish(PublishDecision(report, "test", None))


@pytest.fixture
def registered(filled_settings: Settings, tuning: TuningConfig) -> TuningConfig:
    """Chronos-2 with weather as champion and gradient boosting as challenger."""
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    publish(provider, ChronosModelConfig(context_length=64, use_weather=True), "champion")
    challenger = GradientBoostingModelConfig(max_iter=20, n_lags=3, use_weather=False)
    publish(provider, challenger, "challenger")
    return tuning


@pytest.fixture
def grid_search(filled_settings: Settings, tuning: TuningConfig) -> GridSearchRuns:
    """A small grid search logged to MLflow the way the grid search notebook does it."""
    mlflow.set_experiment("natgas-grid-search")
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    dates = provider.backtest_dates("2024-01-01", "2024-04-30", step_days=30)
    optimizer = GridSearchOptimizer(Backtester(provider, dates))
    models = {
        "naive": {},
        "gradient_boosting": {"grid": {"max_iter": [20], "n_lags": [3, 5], "use_weather": [False]}},
        "chronos": {"grid": {"context_length": [64]}},
    }
    with mlflow.start_run(run_name="grid-search-notebook"):
        mlflow.log_params({"backtest_count": len(dates), "horizon_days": 30})
        optimizer.run(models)
    return GridSearchRuns(tuning.tracking_uri)


def test_trading_calendar_skips_exchange_holidays() -> None:
    calendar = TradingCalendar()
    thanksgiving_week = calendar.next_days(pd.Timestamp("2026-11-25"), 2)
    assert list(thanksgiving_week.date) == [date(2026, 11, 27), date(2026, 11, 30)]
    assert date(2026, 12, 25) not in calendar.next_days(pd.Timestamp("2026-12-20"), 10).date
    assert date(2027, 1, 1) not in calendar.next_days(pd.Timestamp("2026-12-28"), 5).date
    # Good Friday is closed, Columbus Day is open
    assert date(2026, 4, 3) not in calendar.next_days(pd.Timestamp("2026-04-01"), 3).date
    assert date(2026, 10, 12) in calendar.next_days(pd.Timestamp("2026-10-09"), 1).date


def test_forecast_dates_past_the_history_skip_holidays() -> None:
    dates = pd.bdate_range("2026-01-02", "2026-11-25")
    provider = ForecastDataProvider(pd.Series(3.0, index=dates), horizon_days=3)
    expected = [date(2026, 11, 27), date(2026, 11, 30), date(2026, 12, 1)]
    assert list(provider.forecast_dates(dates[-1]).date) == expected


def test_live_weather_uses_the_forecast_then_normals() -> None:
    days = pd.date_range("2014-01-01", "2024-06-30")
    hdd = 15 + 15 * np.cos(2 * np.pi * days.dayofyear / 365.25)
    weather = pd.DataFrame({"date": days, "location": "chicago", "hdd": hdd, "cdd": 0.0})
    source = ForecastWeatherFeatureSource(weather)
    as_of = pd.Timestamp("2024-06-20")
    dates = pd.date_range("2024-06-18", "2024-08-15")
    features = source.features_for(dates, as_of)

    assert features.notna().all().all()
    stored = dates[dates <= "2024-06-30"]
    assert np.allclose(features.loc[stored, "hdd"], hdd[days.get_indexer(stored)])
    # Past the forecast the normals follow the yearly cycle
    later = features.loc["2024-07-01":, "hdd"]
    expected = 15 + 15 * np.cos(2 * np.pi * later.index.dayofyear / 365.25)
    assert np.abs(later.to_numpy() - expected).max() < 0.5

    kinds = source.kinds_for(dates, as_of)
    assert kinds.loc[:"2024-06-20"].eq("observed").all()
    assert kinds.loc["2024-06-21":"2024-06-30"].eq("forecast").all()
    assert kinds.loc["2024-07-01":].eq("normal").all()


def test_live_weather_is_only_for_scoring(filled_settings: Settings) -> None:
    backtest = ForecastDataProvider.from_settings(filled_settings, 30)
    live = ForecastDataProvider.from_settings(filled_settings, 30, live_weather=True)
    assert type(backtest.feature_sources[0]) is WeatherFeatureSource
    assert type(live.feature_sources[0]) is ForecastWeatherFeatureSource
    future = live.window(live.last_date).future_features
    assert future[["hdd", "cdd"]].notna().all().all()


def test_chronos_returns_an_interval_and_others_leave_it_empty(filled_settings) -> None:
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    dates = provider.backtest_dates("2024-01-01", "2024-03-31", step_days=60)
    backtester = Backtester(provider, dates)
    chronos = backtester.run(ChronosModelConfig(context_length=64)).forecasts()
    assert (chronos["lower"] < chronos["prediction"]).all()
    assert (chronos["prediction"] < chronos["upper"]).all()
    naive = backtester.run(NaiveModelConfig()).forecasts()
    assert naive[["lower", "upper"]].isna().all().all()


def test_scoring_rules() -> None:
    with pytest.raises(ValueError):
        ProductionScorer.check_rules(ChronosModelConfig(use_weather=False))
    ProductionScorer.check_rules(ChronosModelConfig(use_weather=True))
    ProductionScorer.check_rules(GreykiteModelConfig())
    ProductionScorer.check_rules(GradientBoostingModelConfig(use_weather=False))


def test_scoring_pipeline_end_to_end(filled_settings: Settings, registered: TuningConfig) -> None:
    first = ScoringPipeline(filled_settings, registered, date(2024, 7, 1), refresh=False).run()
    assert first.as_of_date == pd.Timestamp(LAST_PRICE)
    assert first.saved == {"production": 30, "staging": 30}
    assert first.weather_days == {"forecast": 9, "normal": 21}
    rows = read(
        filled_settings,
        "SELECT stage, model, model_version, COUNT(*), COUNT(lower), MIN(target_date), "
        "MAX(horizon_day) FROM predictions GROUP BY stage, model, model_version ORDER BY stage",
    )
    assert rows == [
        ("production", "chronos", "1", 30, 30, "2024-07-01", 30),
        ("staging", "gradient_boosting", "2", 30, 0, "2024-07-01", 30),
    ]

    # A second run for the same day adds nothing
    again = ScoringPipeline(filled_settings, registered, date(2024, 7, 1), refresh=False).run()
    assert again.saved == {"production": 0, "staging": 0}
    assert read(filled_settings, "SELECT COUNT(*) FROM predictions") == [(60,)]

    # Ten days later seven trading days of actuals are in, since July 4 is closed
    new_days = pd.bdate_range("2024-07-01", "2024-07-10").drop(pd.Timestamp("2024-07-04"))
    write_prices(filled_settings, new_days, seed=1)
    later = ScoringPipeline(filled_settings, registered, date(2024, 7, 11), refresh=False).run()
    assert later.as_of_date == pd.Timestamp("2024-07-10")
    assert later.saved == {"production": 30, "staging": 30}
    assert later.actuals == 14

    matured = read(
        filled_settings,
        "SELECT a.forecast - p.high, a.error, a.horizon_bucket, a.horizon_day "
        "FROM forecast_actuals AS a JOIN prices AS p ON p.date = a.target_date",
    )
    assert all(abs(expected - error) < 1e-9 for expected, error, _, _ in matured)
    assert {(bucket, day <= 5) for _, _, bucket, day in matured} == {
        ("days 1 to 5", True),
        ("days 6 to 15", False),
    }

    monitor = ForecastMonitor(filled_settings.db_url)
    accuracy = monitor.accuracy().set_index(["stage", "horizon_bucket"])
    errors = read(filled_settings, "SELECT error FROM forecast_actuals WHERE stage = 'production'")
    expected_mae = np.mean([abs(error) for (error,) in errors])
    assert accuracy.loc[("production", "all days"), "mae"] == pytest.approx(expected_mae)
    assert accuracy.loc[("production", "all days"), "forecasts"] == 7
    assert accuracy.loc[("staging", "days 1 to 5"), "forecasts"] == 5

    rolling = monitor.rolling()
    assert set(rolling["horizon_bucket"]) == {"days 1 to 5", "days 6 to 15", "all days"}
    assert rolling[["mae", "mape", "wape", "bias"]].notna().all().all()


def test_actuals_wait_for_a_complete_day(filled_settings: Settings, registered) -> None:
    ScoringPipeline(filled_settings, registered, date(2024, 7, 1), refresh=False).run()
    write_prices(filled_settings, pd.bdate_range("2024-07-01", periods=1), seed=1)
    refresher = ActualsRefresher(filled_settings.db_url)
    # The July 1 price is only used once July 1 is a complete day
    assert refresher.refresh(date(2024, 6, 28)) == 0
    assert refresher.refresh(date(2024, 7, 1)) == 2
    assert refresher.refresh(date(2024, 7, 1)) == 2
    assert read(filled_settings, "SELECT COUNT(*) FROM forecast_actuals") == [(2,)]


def test_grid_search_configs_round_trip(grid_search: GridSearchRuns) -> None:
    best = grid_search.best_runs(grid_search.latest_search())
    assert set(best) == {"naive", "gradient_boosting", "chronos"}
    config = grid_search.config_for(best["gradient_boosting"])
    assert isinstance(config, GradientBoostingModelConfig)
    assert config.max_iter == 20
    assert config.use_weather is False
    assert config.weather_columns == ["hdd", "cdd"]
    assert GridSearchRuns.parse("0.03", float) == 0.03


def test_backtest_export_and_challenger(
    grid_search: GridSearchRuns, filled_settings: Settings, tmp_path: Path
) -> None:
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    assert ChallengerPublisher(grid_search, provider, MODEL_NAME).publish() == "1"
    assert str(MlflowClient().get_model_version_by_alias(MODEL_NAME, "challenger").version) == "1"

    exporter = BacktestExporter(grid_search, provider, filled_settings.db_url, MODEL_NAME)
    frame = exporter.build()
    as_of_dates = frame["as_of_date"].nunique()
    assert len(frame) == 3 * as_of_dates * 30
    versions = frame.groupby("model")["model_version"].first()
    assert versions["gradient_boosting"] == "1"
    assert versions["naive"].startswith("run-")
    assert frame.loc[frame["model"] == "chronos", "lower"].notna().all()
    assert frame.loc[frame["model"] != "chronos", "lower"].isna().all()

    path = tmp_path / "outputs" / "backtest_forecasts.parquet"
    rows = exporter.write(frame, path)
    assert rows == len(frame)
    assert read(filled_settings, "SELECT COUNT(*) FROM backtest_forecasts") == [(rows,)]
    assert list(pd.read_parquet(path).columns) == list(frame.columns)


def test_export_adds_intervals_only_where_the_rerun_matches(
    grid_search: GridSearchRuns, filled_settings: Settings
) -> None:
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    exporter = BacktestExporter(grid_search, provider, filled_settings.db_url, MODEL_NAME)
    run = grid_search.best_runs(grid_search.latest_search())["chronos"]
    logged = grid_search.forecasts(run).drop(columns=["lower", "upper"])
    first = logged["as_of"].min()
    logged.loc[logged["as_of"] == first, "prediction"] += 1.0

    with_intervals = exporter.add_intervals(logged, run)
    changed = with_intervals["as_of"] == first
    assert with_intervals.loc[changed, "lower"].isna().all()
    assert with_intervals.loc[~changed, "lower"].notna().all()
    assert np.allclose(with_intervals["prediction"], logged["prediction"])

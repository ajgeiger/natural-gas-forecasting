"""Tests for production scoring, the backtest export, actuals, and monitoring. No network."""

from datetime import date, datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sqlalchemy import insert, text

from backtesting import Backtester
from config import Settings
from data import (
    ForecastDataProvider,
    ForecastWeatherFeatureSource,
    TradingCalendar,
    WeatherFeatureSource,
)
from data.database import Prediction, Price, WeatherDaily, get_engine, session_scope, upsert
from models import (
    ChronosModelConfig,
    GradientBoostingModelConfig,
    GreykiteModelConfig,
    NaiveModelConfig,
)
from optimization.config.TuningConfig import TuningConfig
from optimization.optimizers.GridSearchOptimizer import GridSearchOptimizer
from production.publishing.ActualsRefresher import ActualsRefresher
from production.publishing.BacktestExporter import BacktestExporter
from production.publishing.ForecastMonitor import ForecastMonitor
from production.publishing.PublishedForecast import PublishedForecast
from production.registry.GridSearchRuns import GridSearchRuns
from production.registry.ModelPublisher import ModelPublisher
from production.registry.PublishDecision import PublishDecision
from production.scoring.ProductionScorer import ProductionScorer
from production.scoring.ScoringPipeline import ScoringPipeline
from production.tournament.TournamentConfig import TournamentConfig

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


CONTENDER_STAGES = [
    "chronos:champion",
    "chronos:challenger",
    "gradient_boosting:champion",
    "gradient_boosting:challenger",
    "baseline:naive",
]


def publish(
    provider: ForecastDataProvider, config, alias: str, registered_model_name: str = MODEL_NAME
) -> str:
    dates = provider.backtest_dates("2024-01-01", "2024-04-30", step_days=60)
    report = Backtester(provider, dates).run(config)
    with mlflow.start_run():
        publisher = ModelPublisher(provider, registered_model_name, alias)
        return publisher.publish(PublishDecision(report, "test", None))


@pytest.fixture
def tournament(tuning: TuningConfig) -> TournamentConfig:
    return TournamentConfig(
        families=["chronos", "gradient_boosting"], tracking_uri=tuning.tracking_uri
    )


@pytest.fixture
def registered(
    filled_settings: Settings, tuning: TuningConfig, tournament: TournamentConfig
) -> TuningConfig:
    """A champion and a challenger for Chronos-2 and gradient boosting, and naive."""
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    contenders = [
        ("chronos", "champion", ChronosModelConfig(context_length=64, use_weather=True)),
        ("chronos", "challenger", ChronosModelConfig(context_length=128, use_weather=True)),
        (
            "gradient_boosting",
            "champion",
            GradientBoostingModelConfig(max_iter=20, n_lags=3, use_weather=False),
        ),
        (
            "gradient_boosting",
            "challenger",
            GradientBoostingModelConfig(max_iter=20, n_lags=5, use_weather=False),
        ),
        ("naive", "baseline", NaiveModelConfig()),
    ]
    for family, role, config in contenders:
        publish(provider, config, role, tournament.registered_model(family))
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


def pipeline(settings, tuning, tournament, today: date) -> ScoringPipeline:
    return ScoringPipeline(settings, tuning, today, refresh=False, tournament=tournament)


def test_scoring_pipeline_end_to_end(
    filled_settings: Settings, registered: TuningConfig, tournament: TournamentConfig
) -> None:
    first = pipeline(filled_settings, registered, tournament, date(2024, 7, 1)).run()
    assert first.as_of_date == pd.Timestamp(LAST_PRICE)
    assert first.saved == {stage: 30 for stage in CONTENDER_STAGES}
    assert first.weather_days == {"forecast": 9, "normal": 21}
    assert set(first.events) == {"days 1 to 5", "days 6 to 15", "days 16 to 30"}
    assert first.stitched == 30
    rows = read(
        filled_settings,
        "SELECT stage, model, model_version, COUNT(*), COUNT(lower), MIN(target_date), "
        "MAX(horizon_day) FROM predictions WHERE stage != 'production' "
        "GROUP BY stage, model, model_version ORDER BY stage",
    )
    assert rows == [
        ("baseline:naive", "naive", "1", 30, 0, "2024-07-01", 30),
        ("chronos:challenger", "chronos", "2", 30, 30, "2024-07-01", 30),
        ("chronos:champion", "chronos", "1", 30, 30, "2024-07-01", 30),
        ("gradient_boosting:challenger", "gradient_boosting", "2", 30, 0, "2024-07-01", 30),
        ("gradient_boosting:champion", "gradient_boosting", "1", 30, 0, "2024-07-01", 30),
    ]

    # Every published row is a copy of what the event winner forecast in its own stage
    published = read(
        filled_settings,
        "SELECT p.horizon_day, p.model, p.model_version, w.stage, p.forecast, c.forecast "
        "FROM predictions AS p "
        "JOIN event_winners AS w ON w.as_of_date = p.as_of_date "
        "AND p.horizon_day BETWEEN w.first_day AND w.last_day "
        "JOIN predictions AS c ON c.as_of_date = p.as_of_date AND c.stage = w.stage "
        "AND c.horizon_day = p.horizon_day "
        "WHERE p.stage = 'production'",
    )
    assert len(published) == 30
    assert all(forecast == copied for *_, forecast, copied in published)

    # A second run for the same day adds nothing
    again = pipeline(filled_settings, registered, tournament, date(2024, 7, 1)).run()
    assert again.saved == {stage: 0 for stage in CONTENDER_STAGES}
    assert again.stitched == 0
    assert read(filled_settings, "SELECT COUNT(*) FROM predictions") == [(180,)]

    # Ten days later seven trading days of actuals are in, since July 4 is closed
    new_days = pd.bdate_range("2024-07-01", "2024-07-10").drop(pd.Timestamp("2024-07-04"))
    write_prices(filled_settings, new_days, seed=1)
    later = pipeline(filled_settings, registered, tournament, date(2024, 7, 11)).run()
    assert later.as_of_date == pd.Timestamp("2024-07-10")
    assert later.saved == {stage: 30 for stage in CONTENDER_STAGES}
    assert later.actuals == 6 * 7

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
    for stage in CONTENDER_STAGES:
        assert accuracy.loc[(stage, "days 1 to 5"), "forecasts"] == 5
    assert accuracy.loc[("chronos:champion", "all days"), "model"] == "chronos"

    rolling = monitor.rolling()
    assert set(rolling["horizon_bucket"]) == {"days 1 to 5", "days 6 to 15", "all days"}
    assert set(rolling["stage"]) == {"production", *CONTENDER_STAGES}
    assert rolling[["mae", "mape", "wape", "bias"]].notna().all().all()

    # One matured as-of date now counts for 1/30 of each blended score
    standings = read(
        filled_settings,
        "SELECT horizon_bucket, prod_days, prod_weight FROM tournament_standings "
        "WHERE as_of_date = '2024-07-10' AND stage = 'chronos:champion'",
    )
    weights = {bucket: (days, weight) for bucket, days, weight in standings}
    assert weights["days 1 to 5"] == (1, pytest.approx(1 / 30))
    assert weights["days 16 to 30"] == (0, 0.0)


def test_actuals_wait_for_a_complete_day(
    filled_settings: Settings, registered: TuningConfig, tournament: TournamentConfig
) -> None:
    pipeline(filled_settings, registered, tournament, date(2024, 7, 1)).run()
    write_prices(filled_settings, pd.bdate_range("2024-07-01", periods=1), seed=1)
    refresher = ActualsRefresher(filled_settings.db_url)
    # The July 1 price is only used once July 1 is a complete day
    assert refresher.refresh(date(2024, 6, 28)) == 0
    assert refresher.refresh(date(2024, 7, 1)) == 6
    assert refresher.refresh(date(2024, 7, 1)) == 6
    assert read(filled_settings, "SELECT COUNT(*) FROM forecast_actuals") == [(6,)]


def test_grid_search_configs_round_trip(grid_search: GridSearchRuns) -> None:
    best = grid_search.best_runs(grid_search.latest_search())
    assert set(best) == {"naive", "gradient_boosting", "chronos"}
    config = grid_search.config_for(best["gradient_boosting"])
    assert isinstance(config, GradientBoostingModelConfig)
    assert config.max_iter == 20
    assert config.use_weather is False
    assert config.weather_columns == ["hdd", "cdd"]
    assert GridSearchRuns.parse("0.03", float) == 0.03


def test_backtest_export(
    grid_search: GridSearchRuns, filled_settings: Settings, tmp_path: Path
) -> None:
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    search = grid_search.latest_search()
    run = grid_search.best_runs(search)["gradient_boosting"]
    with mlflow.start_run():
        mlflow.set_tags({"search_run_id": search.info.run_id})
        publisher = ModelPublisher(provider, MODEL_NAME, "champion")
        publisher.publish_config(grid_search.config_for(run), {"mae": 1.0}, "test")
    assert str(MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version) == "1"

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


def test_published_forecast_returns_the_newest_production_rows(settings: Settings) -> None:
    run_time = datetime(2024, 6, 28, tzinfo=timezone.utc)
    rows = []
    for as_of, value in ((date(2024, 6, 27), 2.5), (date(2024, 6, 28), 3.5)):
        for day in range(1, 31):
            rows.append(
                {
                    "as_of_date": as_of,
                    "stage": "production",
                    "horizon_day": day,
                    "run_id": "r" * 32,
                    "run_timestamp": run_time,
                    "target_date": as_of,
                    "model": "tournament",
                    "model_version": "1",
                    "forecast": value + 0.01 * day,
                    "lower": value,
                    "upper": value + 1,
                }
            )
    rows.append(
        {
            "as_of_date": date(2024, 6, 28),
            "stage": "chronos:champion",
            "horizon_day": 1,
            "run_id": "r" * 32,
            "run_timestamp": run_time,
            "target_date": date(2024, 6, 28),
            "model": "chronos",
            "model_version": "1",
            "forecast": 9.9,
            "lower": None,
            "upper": None,
        }
    )
    with session_scope(settings.db_url) as session:
        upsert(
            session,
            Price,
            [
                {"date": date(2024, 6, 20), "high": 2.0, "source": "test"},
                {"date": date(2024, 6, 28), "high": 3.2, "source": "test"},
            ],
        )
        session.execute(insert(Prediction), rows)

    published = PublishedForecast(settings.db_url)
    forecast = published.latest()
    assert len(forecast) == 30
    assert set(pd.to_datetime(forecast["as_of_date"]).dt.date) == {date(2024, 6, 28)}
    assert forecast["forecast"].iloc[0] == pytest.approx(3.51)
    assert list(forecast["horizon_day"]) == list(range(1, 31))

    latest_price = published.prices(days=1)
    assert list(latest_price["high"]) == [pytest.approx(3.2)]
    recent = published.prices(days=10)
    assert len(recent) == 2
    assert list(pd.to_datetime(recent["date"]).dt.date) == [date(2024, 6, 20), date(2024, 6, 28)]

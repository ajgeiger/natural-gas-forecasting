"""Tests for the data provider, models, backtester, optimizers, and publisher. No network."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from backtesting import Backtester, BacktestReport, ForecastMetrics
from data import (
    AbstractFeatureSource,
    CalendarFeatureSource,
    ForecastDataProvider,
    HolidayFeatureSource,
    StorageFeatureSource,
    WeatherFeatureSource,
)
from models import (
    AbstractConfigWrapper,
    AbstractModelWrapper,
    ChronosModel,
    ChronosModelConfig,
    GradientBoostingModelConfig,
    GreykiteModel,
    GreykiteModelConfig,
    ModelCatalog,
    NaiveModel,
    NaiveModelConfig,
)
from optimization.config.TuningConfig import TuningConfig
from optimization.optimizers.BayesianOptimizer import BayesianOptimizer
from optimization.optimizers.GridSearchOptimizer import GridSearchOptimizer
from optimization.optimizers.OptimizerCatalog import OptimizerCatalog
from optimization.pipeline.TuningPipeline import TuningPipeline
from optimization.search_space.HyperparameterGrid import HyperparameterGrid
from production.registry.ModelPublisher import ModelPublisher


@pytest.fixture(scope="module")
def holiday_source() -> HolidayFeatureSource:
    return HolidayFeatureSource()


@pytest.fixture
def provider(holiday_source: HolidayFeatureSource) -> ForecastDataProvider:
    dates = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(0)
    target = pd.Series(3 + np.cumsum(rng.normal(0, 0.05, len(dates))), index=dates)
    weather_days = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=14))
    weather = pd.DataFrame(
        {
            "date": weather_days,
            "location": "chicago",
            "hdd": rng.uniform(0, 30, len(weather_days)),
            "cdd": rng.uniform(0, 10, len(weather_days)),
        }
    )
    sources = [WeatherFeatureSource(weather), holiday_source, CalendarFeatureSource()]
    return ForecastDataProvider(target, sources, horizon_days=5)


def test_abstract_classes_cannot_be_created() -> None:
    abstract_classes = (AbstractModelWrapper, AbstractConfigWrapper, AbstractFeatureSource)
    for abstract_class in abstract_classes:
        with pytest.raises(TypeError):
            abstract_class()


def test_window_has_no_training_rows_after_as_of(provider: ForecastDataProvider) -> None:
    for as_of in provider.backtest_dates("2022-06-01", step_days=20):
        window = provider.window(as_of)
        assert window.training_data.index.max() <= as_of
        assert window.forecast_dates.min() > as_of
        assert len(window.forecast_dates) == 5
        assert window.test_target.notna().all()


def test_weather_after_as_of_can_be_hidden() -> None:
    dates = pd.date_range("2024-01-01", periods=10)
    weather = pd.DataFrame({"date": dates, "location": "x", "hdd": 1.0, "cdd": 2.0})
    source = WeatherFeatureSource(weather, use_observed_as_forecast=False)
    frame = source.features_for(dates, pd.Timestamp("2024-01-05"))
    assert frame.loc[:"2024-01-05"].notna().all().all()
    assert frame.loc["2024-01-06":].isna().all().all()


def test_storage_waits_for_release_date() -> None:
    weekly = pd.DataFrame({"week_ending": ["2024-01-05", "2024-01-12"], "value_bcf": [100.0, 90.0]})
    source = StorageFeatureSource(weekly)
    dates = pd.date_range("2024-01-10", "2024-01-20")
    frame = source.features_for(dates, as_of=pd.Timestamp("2024-01-15"))
    # Week ending Jan 5 is released Jan 11, week ending Jan 12 on Jan 18
    assert np.isnan(frame.loc["2024-01-10", "storage_bcf"])
    assert frame.loc["2024-01-11", "storage_bcf"] == 100.0
    # Jan 18 is after the as-of date, so the Jan 12 report is not known yet
    assert frame.loc["2024-01-20", "storage_bcf"] == 100.0


def test_metrics() -> None:
    scores = ForecastMetrics.score(pd.Series([2.0, 4.0]), pd.Series([3.0, 3.0]))
    assert scores["mae"] == 1.0
    assert scores["rmse"] == 1.0
    assert scores["mape"] == pytest.approx(37.5)
    assert scores["wape"] == pytest.approx(100 * 2 / 6)
    assert scores["bias"] == 0.0


def test_grid_expands_every_combination() -> None:
    configs = HyperparameterGrid({"a": [1, 2], "b": [True, False], "c": 5}).configs()
    assert len(configs) == 4
    assert {"a": 2, "b": False, "c": 5} in configs


def test_configs_validate_and_round_trip() -> None:
    config = GradientBoostingModelConfig(max_depth=4, use_weather=False)
    assert GradientBoostingModelConfig.from_dict(config.to_dict()) == config
    assert config.required_features == []
    assert GreykiteModelConfig().required_features == HolidayFeatureSource.column_names(4)
    assert GreykiteModelConfig(use_holidays=False).required_features == []
    assert ChronosModelConfig().required_features == ["hdd", "cdd"]
    assert ChronosModelConfig().max_horizon_days == 1024
    calendar = GradientBoostingModelConfig(
        use_weather=False, use_yearly=True, weekly_harmonics=1, use_weekend=True
    )
    assert calendar.required_features == [
        "year_sin_1",
        "year_cos_1",
        "year_sin_2",
        "year_cos_2",
        "week_sin_1",
        "week_cos_1",
        "after_weekend",
    ]
    assert ChronosModelConfig(weekly_harmonics=2).required_features == [
        "hdd",
        "cdd",
        "week_sin_1",
        "week_cos_1",
        "week_sin_2",
        "week_cos_2",
    ]
    with pytest.raises(ValueError):
        GradientBoostingModelConfig(weekly_harmonics=3)
    with pytest.raises(ValueError):
        ChronosModelConfig(weekly_harmonics=-1)
    with pytest.raises(ValueError):
        GradientBoostingModelConfig.from_dict({"not_a_setting": 1})
    with pytest.raises(ValueError):
        ChronosModelConfig(context_length=15)
    with pytest.raises(ValueError):
        ChronosModelConfig(model_id="not-a-chronos-model")
    with pytest.raises(ValueError):
        GreykiteModelConfig(fit_algorithm="lasso")
    with pytest.raises(ValueError):
        GreykiteModelConfig(holiday_window=5)


def test_catalog_pairs_every_model_with_its_config() -> None:
    for name in ModelCatalog.names():
        model_class, config_class = ModelCatalog.entry(name)
        assert model_class.config_class is config_class
        assert config_class.model_name == name
        for values in HyperparameterGrid(config_class.default_grid()).configs():
            config_class.from_dict(values)


def test_model_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError):
        NaiveModel(GradientBoostingModelConfig())


@pytest.mark.parametrize(
    "config",
    [
        NaiveModelConfig(),
        GradientBoostingModelConfig(max_iter=20),
        GreykiteModelConfig(history_years=0),
        ChronosModelConfig(context_length=64),
    ],
)
def test_models_train_predict_save_load(
    config: AbstractConfigWrapper, provider: ForecastDataProvider, tmp_path: Path
) -> None:
    window = provider.window(provider.last_date - pd.Timedelta(days=30))
    model = ModelCatalog.build(config)
    model.train(window.training_data)
    prediction = model.predict(window.future_features)
    assert prediction.index.equals(window.forecast_dates)
    assert prediction.notna().all()

    model.save(tmp_path / "model")
    loaded = ModelCatalog.load(tmp_path / "model")
    assert type(loaded) is type(model)
    assert loaded.config == config
    pd.testing.assert_series_equal(loaded.predict(window.future_features), prediction)


def test_holiday_columns_follow_the_window() -> None:
    columns = HolidayFeatureSource.column_names(4)
    labor_day = [column for column in columns if column.startswith("labor_day")]
    assert labor_day == [
        "labor_day",
        "labor_day_pre_1",
        "labor_day_pre_2",
        "labor_day_pre_3",
        "labor_day_pre_4",
        "labor_day_post_1",
        "labor_day_post_2",
        "labor_day_post_3",
        "labor_day_post_4",
    ]
    assert "labor_day_pre_3" not in HolidayFeatureSource.column_names(2)
    assert "labor_day_pre_1" not in HolidayFeatureSource.column_names(0)
    assert "new_years_day" in columns
    with pytest.raises(ValueError):
        HolidayFeatureSource(window=5)


def test_holiday_dummies_mark_days_around_each_holiday(
    holiday_source: HolidayFeatureSource,
) -> None:
    # Labor Day 2025 was Monday September 1
    dates = pd.date_range("2025-08-28", "2025-09-05")
    features = holiday_source.features_for(dates, pd.Timestamp("2025-08-27"))
    assert features.loc["2025-09-01", "labor_day"] == 1
    assert features.loc["2025-09-02", "labor_day_post_1"] == 1
    assert features.loc["2025-08-28", "labor_day_pre_4"] == 1
    assert features.loc["2025-09-05", "labor_day_post_4"] == 1
    assert features["labor_day"].sum() == 1
    # Observed days count as the holiday: July 4 2026 is a Saturday
    observed = holiday_source.features_for(pd.DatetimeIndex(["2026-07-03"]), dates[0])
    assert observed.loc["2026-07-03", "independence_day"] == 1


def test_calendar_columns_follow_the_settings() -> None:
    assert CalendarFeatureSource().columns == [
        "year_sin_1",
        "year_cos_1",
        "year_sin_2",
        "year_cos_2",
        "week_sin_1",
        "week_cos_1",
        "week_sin_2",
        "week_cos_2",
        "after_weekend",
    ]
    assert CalendarFeatureSource.column_names(False, 1, False) == ["week_sin_1", "week_cos_1"]
    assert CalendarFeatureSource.column_names(False, 0, False) == []


def test_calendar_values_for_known_dates() -> None:
    # Saturday August 30 to Tuesday September 2, 2025
    dates = pd.date_range("2025-08-30", "2025-09-02")
    features = CalendarFeatureSource().features_for(dates, dates[0])
    assert features["after_weekend"].tolist() == [0.0, 1.0, 1.0, 0.0]
    monday = features.loc["2025-09-01"]
    assert monday["week_sin_1"] == pytest.approx(0.0)
    assert monday["week_cos_1"] == pytest.approx(1.0)
    assert monday["year_sin_1"] == pytest.approx(np.sin(2 * np.pi * 244 / 365.25))
    assert monday["year_cos_2"] == pytest.approx(np.cos(4 * np.pi * 244 / 365.25))
    tuesday = features.loc["2025-09-02"]
    assert tuesday["week_sin_2"] == pytest.approx(np.sin(4 * np.pi / 7))


def test_calendar_features_cover_future_dates(provider: ForecastDataProvider) -> None:
    window = provider.window(provider.last_date)
    columns = CalendarFeatureSource().columns
    future = window.future_features[columns]
    assert window.forecast_dates.min() > window.as_of
    assert future.notna().all().all()
    expected = CalendarFeatureSource().features_for(window.forecast_dates, window.as_of)
    pd.testing.assert_frame_equal(future, expected)


def test_holiday_dummies_cover_future_dates(provider: ForecastDataProvider) -> None:
    window = provider.window(provider.last_date)
    columns = HolidayFeatureSource.column_names(4)
    assert window.forecast_dates.min() > window.as_of
    assert window.future_features[columns].notna().all().all()


def test_greykite_uses_holiday_regressors(provider: ForecastDataProvider) -> None:
    # The history starts in January 2022, so Christmas has not happened yet
    window = provider.window(pd.Timestamp("2022-12-01"))
    model = GreykiteModel(GreykiteModelConfig(autoregression_lags=0, holiday_window=2))
    model.train(window.training_data)
    assert list(model.history.columns) == ["price", *HolidayFeatureSource.column_names(2)]
    regressors = model.regressor_columns()
    assert regressors
    assert (model.history[regressors].sum() > 0).all()
    assert "christmas_day" not in regressors
    assert "labor_day_post_1" in regressors
    prediction = model.predict(window.future_features)
    assert prediction.index.equals(window.forecast_dates)
    assert prediction.notna().all()

    without_holidays = GreykiteModel(GreykiteModelConfig(autoregression_lags=0, use_holidays=False))
    without_holidays.train(window.training_data)
    assert list(without_holidays.history.columns) == ["price"]


def test_chronos_uses_weather_covariates(provider: ForecastDataProvider) -> None:
    window = provider.window(provider.last_date - pd.Timedelta(days=30))
    with_weather = ChronosModel(ChronosModelConfig(context_length=64, use_weather=True))
    with_weather.train(window.training_data)
    assert list(with_weather.context.columns) == ["price", "hdd", "cdd"]
    assert with_weather.context.index.max() <= window.as_of
    future_calendar = pd.bdate_range(window.as_of + pd.Timedelta(days=1), window.forecast_dates[-1])
    covariates = with_weather.future_covariates(window.future_features, future_calendar)
    assert covariates["timestamp"].min() > window.as_of
    assert covariates[["hdd", "cdd"]].notna().all().all()
    prediction = with_weather.predict(window.future_features)
    assert prediction.index.equals(window.forecast_dates)
    assert prediction.notna().all()

    without_weather = ChronosModel(ChronosModelConfig(context_length=64, use_weather=False))
    without_weather.train(window.training_data)
    assert list(without_weather.context.columns) == ["price"]


def test_backtester_records_each_as_of(provider: ForecastDataProvider) -> None:
    dates = provider.backtest_dates("2022-09-01", step_days=30)
    report = Backtester(provider, dates).run(NaiveModelConfig())
    assert [result.as_of for result in report.results] == dates
    assert all(result.last_training_date <= result.as_of for result in report.results)
    assert len(report.forecasts()) == 5 * len(dates)
    assert set(report.metrics()) == set(ForecastMetrics.names)


@pytest.mark.parametrize(
    "config",
    [
        NaiveModelConfig(),
        GradientBoostingModelConfig(max_iter=20, use_weather=True),
        GradientBoostingModelConfig(
            max_iter=20, use_yearly=True, weekly_harmonics=2, use_weekend=True
        ),
        GreykiteModelConfig(history_years=0, fit_algorithm="ridge"),
        ChronosModelConfig(context_length=64, use_weather=True),
        ChronosModelConfig(
            context_length=64, use_yearly=True, weekly_harmonics=1, use_weekend=True
        ),
    ],
)
def test_models_forecast_30_days_ahead(
    config: AbstractConfigWrapper, provider: ForecastDataProvider
) -> None:
    long_range = ForecastDataProvider(provider.target, provider.feature_sources, horizon_days=30)
    dates = long_range.backtest_dates("2023-01-05", "2023-05-15", step_days=60)
    report = Backtester(long_range, dates).run(config)
    forecasts = report.forecasts()
    assert len(forecasts) == 30 * len(dates)
    assert forecasts["step"].max() == 30
    assert forecasts["prediction"].notna().all()
    buckets = report.metrics_by_bucket()
    assert list(buckets.index) == ["days 1 to 5", "days 6 to 15", "days 16 to 30"]
    assert list(buckets.columns) == ForecastMetrics.names


def test_buckets_past_the_horizon_are_skipped(provider: ForecastDataProvider) -> None:
    dates = provider.backtest_dates("2022-09-01", step_days=30)
    report = Backtester(provider, dates).run(NaiveModelConfig())
    assert list(report.metrics_by_bucket().index) == ["days 1 to 5"]
    custom = report.metrics_by_bucket([(1, 2), (3, 5)])
    assert list(custom.index) == ["days 1 to 2", "days 3 to 5"]


def test_backtester_checks_config_against_provider(provider: ForecastDataProvider) -> None:
    no_weather = ForecastDataProvider(provider.target, [], horizon_days=5)
    backtester = Backtester(no_weather, no_weather.backtest_dates("2022-09-01", step_days=30))
    with pytest.raises(ValueError):
        backtester.run(GradientBoostingModelConfig(use_weather=True))


def test_publisher_keeps_naive_unless_margin_is_cleared(provider: ForecastDataProvider) -> None:
    dates = provider.backtest_dates("2023-01-01", step_days=30)
    backtester = Backtester(provider, dates)
    naive = backtester.run(NaiveModelConfig())
    chronos = backtester.run(ChronosModelConfig(context_length=64))
    reports: list[BacktestReport] = [naive, chronos]
    improvement = 1 - min(r.metrics()["mae"] for r in reports) / naive.metrics()["mae"]

    strict = ModelPublisher(provider, "x", min_improvement=improvement + 0.5)
    assert strict.choose(reports).report.model_name == "naive"
    if improvement > 0:
        loose = ModelPublisher(provider, "x", min_improvement=0.0)
        assert loose.choose(reports).report.model_name == "chronos"


def run_pipeline(provider: ForecastDataProvider, tmp_path: Path, optimizer: str):
    config = TuningConfig(
        model_settings={
            "naive": {},
            "gradient_boosting": {
                "grid": {"max_depth": [2, 3], "max_iter": [20]},
                "search_space": {"max_depth": {"low": 2, "high": 4}, "max_iter": [20]},
            },
            "chronos": {"grid": {"context_length": [64]}, "search_space": {"context_length": [64]}},
        },
        optimizer=optimizer,
        n_trials=2,
        backtest_start=pd.Timestamp("2023-01-01"),
        step_days=30,
        min_improvement=10.0,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name=f"test-{optimizer}",
        registered_model_name=f"test-{optimizer}-forecaster",
    )
    return TuningPipeline(config, provider).run()


@pytest.mark.parametrize("optimizer", ["grid", "bayesian"])
def test_pipeline_logs_runs_and_publishes(
    optimizer: str, provider: ForecastDataProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    outcome = run_pipeline(provider, tmp_path, optimizer)

    assert outcome.summary["mae"].is_monotonic_increasing
    assert set(outcome.summary["model_name"]) == {"naive", "gradient_boosting", "chronos"}
    assert (outcome.summary["method"] == optimizer).all()
    # An impossible margin means naive is always published
    assert outcome.decision.report.model_name == "naive"
    assert outcome.registered_version == "1"

    experiment = mlflow.get_experiment_by_name(f"test-{optimizer}")
    runs = mlflow.search_runs([experiment.experiment_id])
    # One search run, one run per config, one publish run
    assert len(runs) == len(outcome.summary) + 2

    name = f"test-{optimizer}-forecaster"
    version = MlflowClient().get_model_version_by_alias(name, "champion")
    assert str(version.version) == "1"
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    future = provider.window(provider.last_date).future_features.reset_index(names="date")
    assert len(model.predict(future)) == 5


def test_optimizers_report_their_method(provider: ForecastDataProvider) -> None:
    backtester = Backtester(provider, provider.backtest_dates("2023-01-01", step_days=60))
    assert GridSearchOptimizer(backtester).method_name == "grid"
    assert isinstance(OptimizerCatalog.build("bayesian", backtester), BayesianOptimizer)
    with pytest.raises(KeyError):
        OptimizerCatalog.build("random", backtester)
    assert BayesianOptimizer(backtester, n_trials=3).trial_count({"a": [1, 2]}) == 2

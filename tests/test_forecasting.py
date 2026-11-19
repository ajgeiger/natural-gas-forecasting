"""Tests for the data provider, models, backtester, and tuning pipeline. No network."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from natgas.forecasting.backtesting import Backtester, ForecastMetrics
from natgas.forecasting.data import (
    AbstractFeatureSource,
    ForecastDataProvider,
    StorageFeatureSource,
    WeatherFeatureSource,
)
from natgas.forecasting.models import (
    AbstractModelWrapper,
    GradientBoostingModel,
    ModelCatalog,
    NaiveModel,
)
from natgas.forecasting.tuning import HyperparameterGrid, TuningConfig, TuningPipeline


@pytest.fixture
def provider() -> ForecastDataProvider:
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
    return ForecastDataProvider(target, [WeatherFeatureSource(weather)], horizon_days=5)


def test_abstract_classes_cannot_be_created() -> None:
    with pytest.raises(TypeError):
        AbstractModelWrapper()
    with pytest.raises(TypeError):
        AbstractFeatureSource()


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


@pytest.mark.parametrize("model_class", [NaiveModel, GradientBoostingModel])
def test_models_train_predict_save_load(
    model_class: type[AbstractModelWrapper], provider: ForecastDataProvider, tmp_path: Path
) -> None:
    window = provider.window(provider.last_date - pd.Timedelta(days=30))
    model = model_class()
    model.train(window.training_data)
    prediction = model.predict(window.future_features)
    assert prediction.index.equals(window.forecast_dates)
    assert prediction.notna().all()

    model.save(tmp_path / "model")
    loaded = ModelCatalog.load(tmp_path / "model")
    assert type(loaded) is model_class
    pd.testing.assert_series_equal(loaded.predict(window.future_features), prediction)


def test_unknown_hyperparameter_is_rejected() -> None:
    with pytest.raises(ValueError):
        GradientBoostingModel({"not_a_setting": 1})


def test_backtester_records_each_as_of(provider: ForecastDataProvider) -> None:
    dates = provider.backtest_dates("2022-09-01", step_days=30)
    report = Backtester(provider, dates).run(NaiveModel)
    assert [result.as_of for result in report.results] == dates
    assert all(result.last_training_date <= result.as_of for result in report.results)
    assert len(report.forecasts()) == 5 * len(dates)
    assert set(report.metrics()) == set(ForecastMetrics.names)


def test_pipeline_logs_runs_and_registers_best_model(
    provider: ForecastDataProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = TuningConfig(
        model_grids={"naive": {}, "gradient_boosting": {"max_depth": [2, 3], "max_iter": [20]}},
        backtest_start=pd.Timestamp("2023-01-01"),
        step_days=30,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test-backtests",
        registered_model_name="test-forecaster",
    )
    outcome = TuningPipeline(config, provider).run()

    assert len(outcome.summary) == 3
    assert outcome.summary["mae"].is_monotonic_increasing
    assert outcome.registered_version == "1"

    experiment = mlflow.get_experiment_by_name("test-backtests")
    runs = mlflow.search_runs([experiment.experiment_id])
    # One search run, three config runs, one publish run
    assert len(runs) == 5

    client = MlflowClient()
    version = client.get_model_version_by_alias("test-forecaster", "champion")
    assert str(version.version) == "1"
    model = mlflow.pyfunc.load_model("models:/test-forecaster@champion")
    future = provider.window(provider.last_date).future_features.reset_index(names="date")
    assert len(model.predict(future)) == 5

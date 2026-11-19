"""Tests for the data provider, models, backtester, optimizers, and publisher. No network."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from natgas.forecasting.backtesting import Backtester, BacktestReport, ForecastMetrics
from natgas.forecasting.data import (
    AbstractFeatureSource,
    ForecastDataProvider,
    StorageFeatureSource,
    WeatherFeatureSource,
)
from natgas.forecasting.models import (
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
from natgas.forecasting.optimization import (
    BayesianOptimizer,
    GridSearchOptimizer,
    HyperparameterGrid,
    OptimizerCatalog,
)
from natgas.forecasting.tuning import ModelPublisher, TuningConfig, TuningPipeline


class FakeChronosPipeline:
    """Stands in for a downloaded Chronos-2 model: predicts the last value plus a step."""

    def predict_df(
        self,
        df,
        future_df=None,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=None,
        quantile_levels=None,
        context_length=None,
        freq="B",
        **kwargs,
    ):
        last = float(df[target].iloc[-1])
        start = pd.Timestamp(df[timestamp_column].iloc[-1]) + pd.Timedelta(days=1)
        dates = pd.bdate_range(start, periods=prediction_length, freq=freq or "B")
        values = [last + 0.01 * (k + 1) for k in range(prediction_length)]
        frame = pd.DataFrame(
            {
                id_column: df[id_column].iloc[0],
                timestamp_column: dates,
                "target_name": target,
                "predictions": values,
            }
        )
        for level in quantile_levels or [0.5]:
            frame[str(level)] = values
        return frame


@pytest.fixture(autouse=True)
def fake_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    key = f"{ChronosModelConfig().model_id}@cpu"
    monkeypatch.setattr(ChronosModel, "pipelines", {key: FakeChronosPipeline()})


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
    assert GreykiteModelConfig().required_features == ["hdd", "cdd"]
    assert ChronosModelConfig().required_features == ["hdd", "cdd"]
    assert ChronosModelConfig().max_horizon_days == 1024
    with pytest.raises(ValueError):
        GradientBoostingModelConfig.from_dict({"not_a_setting": 1})
    with pytest.raises(ValueError):
        ChronosModelConfig(context_length=15)
    with pytest.raises(ValueError):
        ChronosModelConfig(model_id="not-a-chronos-model")
    with pytest.raises(ValueError):
        GreykiteModelConfig(fit_algorithm="lasso")


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


def test_greykite_uses_weather_regressors(provider: ForecastDataProvider) -> None:
    window = provider.window(provider.last_date - pd.Timedelta(days=30))
    model = GreykiteModel(GreykiteModelConfig(autoregression_lags=0))
    model.train(window.training_data)
    assert list(model.history.columns) == ["price", "hdd", "cdd"]


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

"""Tests for the model tournament: registries, scorecard, trials, events, and publishing."""

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from conftest import FakeChronosPipeline
from mlflow import MlflowClient
from sqlalchemy import insert, inspect, text
from typer.testing import CliRunner

from natgas import cli
from natgas.config import Settings
from natgas.db import (
    ForecastActual,
    Prediction,
    Price,
    WeatherDaily,
    get_engine,
    init_db,
    session_scope,
    upsert,
)
from natgas.forecasting.backtesting import Backtester
from natgas.forecasting.data import ForecastDataProvider
from natgas.forecasting.models import (
    ChronosModel,
    ChronosModelConfig,
    GradientBoostingModelConfig,
    GreykiteModelConfig,
    NaiveModelConfig,
)
from natgas.forecasting.optimization import GridSearchOptimizer
from natgas.forecasting.production import GridSearchRuns, PredictionStore, StageColumnMigrator
from natgas.forecasting.production.tournament import (
    BlendedScore,
    BlendedScorer,
    Contender,
    EventJudge,
    FamilyRegistry,
    ForecastStitcher,
    LegacyStageRelabeler,
    ModelRules,
    ModelTournament,
    TournamentConfig,
    TournamentSeeder,
)

RUN_TIME = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
LAST_PRICE = date(2024, 6, 28)

CHRONOS = ChronosModelConfig(use_weather=True)
GRADIENT_BOOSTING = GradientBoostingModelConfig(use_weather=False)
NAIVE = NaiveModelConfig()
BACKTEST = {"days 1 to 5": 0.25, "days 6 to 15": 0.46, "days 16 to 30": 0.66, "all days": 0.52}


def contender(family, role, version, config, backtest=None) -> Contender:
    return Contender(family, role, f"natgas-{family}", str(version), config, backtest or BACKTEST)


def bucket_for(horizon_day: int) -> str:
    if horizon_day <= 5:
        return "days 1 to 5"
    return "days 6 to 15" if horizon_day <= 15 else "days 16 to 30"


def add_forecasts(
    settings: Settings,
    item: Contender,
    as_of_dates: list[date],
    error: float,
    matured: bool = True,
    stage: str | None = None,
    horizon_days: int = 30,
) -> None:
    """Predictions for each as-of date and, when matured, their actuals with a fixed error."""
    predictions, actuals = [], []
    for as_of in as_of_dates:
        for day in range(1, horizon_days + 1):
            row = {
                "as_of_date": as_of,
                "stage": stage or item.stage,
                "horizon_day": day,
                "run_id": "r" * 32,
                "target_date": as_of,
                "model": item.model,
                "model_version": item.version,
                "forecast": 3.0 + error,
                "lower": None,
                "upper": None,
            }
            predictions.append({**row, "run_timestamp": RUN_TIME})
            if matured:
                actuals.append(
                    {**row, "horizon_bucket": bucket_for(day), "actual": 3.0, "error": error}
                )
    with session_scope(settings.db_url) as session:
        session.execute(insert(Prediction), predictions)
        if actuals:
            session.execute(insert(ForecastActual), actuals)


def read(settings: Settings, query: str) -> list:
    with get_engine(settings.db_url).connect() as connection:
        return connection.execute(text(query)).all()


class FakeRegistry:
    """Holds contenders in memory and swaps the titles on promotion, like FamilyRegistry."""

    def __init__(self, contenders: list[Contender]) -> None:
        self.contenders = {item.stage: item for item in contenders}

    def load_all(self) -> list[Contender]:
        return list(self.contenders.values())

    def promote(self, family: str) -> tuple[str, str]:
        champion = self.contenders[f"{family}:champion"]
        challenger = self.contenders[f"{family}:challenger"]
        self.contenders[champion.stage] = replace(challenger, role="champion")
        self.contenders[challenger.stage] = replace(champion, role="challenger")
        return challenger.version, champion.version


def trading_days(count: int, start: str = "2026-01-02") -> list[date]:
    return list(pd.bdate_range(start, periods=count).date)


# Configuration and rules


def test_tournament_config_reads_the_yaml() -> None:
    config = TournamentConfig.from_yaml("config/tuning.yaml")
    assert config.families == ["chronos", "gradient_boosting", "greykite"]
    assert config.baseline == "naive"
    assert config.trial_days == 60
    assert config.window_days == 30
    assert config.min_improvement == 0.01
    assert config.registered_model("gradient_boosting") == "natgas-gradient-boosting"
    assert config.registered_model("naive") == "natgas-naive"
    assert TournamentConfig.metric_name("days 1 to 5") == "mae_days_1_to_5"
    assert TournamentConfig.metric_name("all days") == "mae"
    with pytest.raises(ValueError):
        TournamentConfig(families=["naive"])


def test_domain_rules() -> None:
    assert ModelRules.passes(ChronosModelConfig(use_weather=True))
    assert not ModelRules.passes(ChronosModelConfig(use_weather=False))
    assert ModelRules.problem(ChronosModelConfig(use_weather=False)) == (
        "Chronos-2 must use weather covariates"
    )
    assert ModelRules.passes(GreykiteModelConfig())
    assert ModelRules.passes(GradientBoostingModelConfig(use_weather=True))
    assert ModelRules.passes(GradientBoostingModelConfig(use_weather=False))


def test_contender_stage_names() -> None:
    assert contender("chronos", "champion", 1, CHRONOS).stage == "chronos:champion"
    assert (
        contender("gradient_boosting", "challenger", 2, GRADIENT_BOOSTING).stage
        == "gradient_boosting:challenger"
    )
    assert contender("naive", "baseline", 1, NAIVE).stage == "baseline:naive"


# Blended scorecard


@pytest.mark.parametrize(
    ("prod_days", "weight", "blended"),
    [(0, 0.0, 0.6), (15, 0.5, 0.5), (30, 1.0, 0.4), (45, 1.0, 0.4)],
)
def test_blend_moves_from_backtest_to_production(prod_days, weight, blended) -> None:
    prod_weight, score = BlendedScorer.blend(0.4, prod_days, 0.6, window_days=30)
    assert prod_weight == pytest.approx(weight)
    assert score == pytest.approx(blended)


def test_scorer_with_no_production_days_is_the_backtest(settings: Settings) -> None:
    champion = contender("chronos", "champion", 1, CHRONOS)
    score = BlendedScorer(settings.db_url).score(champion, "days 6 to 15", date(2026, 3, 1))
    assert (score.prod_days, score.prod_mae, score.prod_weight) == (0, 0.0, 0.0)
    assert score.blended_mae == pytest.approx(BACKTEST["days 6 to 15"])
    assert score.backtest_source == "model"


def test_scorer_blends_half_a_window(settings: Settings) -> None:
    champion = contender("chronos", "champion", 1, CHRONOS)
    days = trading_days(15)
    add_forecasts(settings, champion, days, error=0.3)
    score = BlendedScorer(settings.db_url).score(champion, "days 1 to 5", days[-1])
    assert score.prod_days == 15
    assert score.prod_mae == pytest.approx(0.3)
    assert score.prod_weight == pytest.approx(0.5)
    assert score.blended_mae == pytest.approx(0.5 * 0.3 + 0.5 * 0.25)


def test_scorer_keeps_the_last_30_dates_of_this_version(settings: Settings) -> None:
    champion = contender("chronos", "champion", 1, CHRONOS)
    days = trading_days(40)
    add_forecasts(settings, champion, days[:10], error=1.0)
    add_forecasts(settings, champion, days[10:], error=-0.2)
    # Another version and the published stage do not count
    add_forecasts(
        settings, replace(champion, version="9"), days, error=5.0, stage="chronos:challenger"
    )
    add_forecasts(settings, champion, days, error=5.0, stage="production")
    scorer = BlendedScorer(settings.db_url)
    score = scorer.score(champion, "all days", days[-1])
    assert score.prod_days == 30
    assert score.prod_mae == pytest.approx(0.2)
    assert score.blended_mae == pytest.approx(0.2)
    # Only dates up to the as-of date count
    earlier = scorer.score(champion, "all days", days[14])
    assert earlier.prod_days == 15
    assert earlier.prod_mae == pytest.approx((10 * 1.0 + 5 * 0.2) / 15)


def test_scorer_counts_a_promoted_challenger_record(settings: Settings) -> None:
    challenger = contender("chronos", "challenger", 2, CHRONOS)
    days = trading_days(6)
    add_forecasts(settings, challenger, days, error=0.1)
    promoted = replace(challenger, role="champion")
    score = BlendedScorer(settings.db_url).score(promoted, "all days", days[-1])
    assert score.prod_days == 6


def test_scorer_falls_back_to_the_naive_backtest(settings: Settings) -> None:
    champion = contender("greykite", "champion", 1, GreykiteModelConfig(), {"all days": 0.7})
    scorer = BlendedScorer(settings.db_url)
    score = scorer.score(champion, "days 1 to 5", date(2026, 1, 2), {"days 1 to 5": 0.24})
    assert score.backtest_source == "naive"
    assert score.blended_mae == pytest.approx(0.24)
    with pytest.raises(ValueError, match="no backtest MAE"):
        scorer.score(champion, "days 1 to 5", date(2026, 1, 2))


# Events


def score_of(item: Contender, bucket: str, blended: float) -> BlendedScore:
    return BlendedScore(
        item.stage, item.model, item.version, bucket, 0, 0.0, blended, "model", 0.0, blended
    )


def test_event_goes_to_the_best_qualified_champion() -> None:
    naive = contender("naive", "baseline", 1, NAIVE)
    chronos = contender("chronos", "champion", 1, CHRONOS)
    boosting = contender("gradient_boosting", "champion", 1, GRADIENT_BOOSTING)
    bucket = "days 1 to 5"
    event = EventJudge(0.01).judge(
        bucket,
        1,
        5,
        [(chronos, score_of(chronos, bucket, 0.496)), (boosting, score_of(boosting, bucket, 0.49))],
        (naive, score_of(naive, bucket, 0.50)),
    )
    # Chronos-2 is only 0.8 percent better than naive, short of the 1 percent margin
    assert event.winner.stage == "gradient_boosting:champion"
    assert event.qualified == ["gradient_boosting:champion"]
    assert event.improvement == pytest.approx(0.02)


def test_event_falls_back_to_naive_when_nobody_qualifies() -> None:
    naive = contender("naive", "baseline", 1, NAIVE)
    chronos = contender("chronos", "champion", 1, CHRONOS)
    bucket = "days 16 to 30"
    event = EventJudge(0.01).judge(
        bucket,
        16,
        30,
        [(chronos, score_of(chronos, bucket, 0.66))],
        (naive, score_of(naive, bucket, 0.65)),
    )
    assert event.winner.stage == "baseline:naive"
    assert event.qualified == []
    assert event.improvement == 0.0
    assert "naive is published" in event.reason


def test_rule_breaker_cannot_qualify() -> None:
    naive = contender("naive", "baseline", 1, NAIVE)
    no_weather = contender("chronos", "champion", 1, ChronosModelConfig(use_weather=False))
    bucket = "days 6 to 15"
    event = EventJudge(0.01).judge(
        bucket,
        6,
        15,
        [(no_weather, score_of(no_weather, bucket, 0.1))],
        (naive, score_of(naive, bucket, 0.5)),
    )
    assert event.winner.stage == "baseline:naive"


# Publishing


def test_stitcher_publishes_each_event_winner(settings: Settings) -> None:
    today = date(2026, 11, 18)
    naive = contender("naive", "baseline", 1, NAIVE)
    chronos = contender("chronos", "champion", 3, CHRONOS)
    boosting = contender("gradient_boosting", "champion", 2, GRADIENT_BOOSTING)
    add_forecasts(settings, naive, [today], 0.0, matured=False)
    add_forecasts(settings, chronos, [today], 0.2, matured=False)
    # Gradient boosting only forecast 10 days, the baseline fills the rest of its event
    add_forecasts(settings, boosting, [today], 0.1, matured=False, horizon_days=10)

    def event(item, first, last, bucket):
        score = score_of(item, bucket, 0.4)
        return SimpleNamespace(
            winner=item, first_day=first, last_day=last, horizon_bucket=bucket, winner_score=score
        )

    events = [
        event(boosting, 1, 5, "days 1 to 5"),
        event(boosting, 6, 15, "days 6 to 15"),
        event(chronos, 16, 30, "days 16 to 30"),
    ]
    stitcher = ForecastStitcher(PredictionStore(settings.db_url))
    assert stitcher.stitch(today, events, naive, "p" * 32, RUN_TIME) == 30
    rows = read(
        settings,
        "SELECT horizon_day, model, model_version, forecast FROM predictions "
        "WHERE stage = 'production' ORDER BY horizon_day",
    )
    assert [row[1] for row in rows] == ["gradient_boosting"] * 10 + ["naive"] * 5 + ["chronos"] * 15
    assert rows[0][2:] == ("2", pytest.approx(3.1))
    assert rows[29][2:] == ("3", pytest.approx(3.2))
    # Production is written once per as-of date
    assert stitcher.stitch(today, events, naive, "p" * 32, RUN_TIME) == 0


# Trials and the full round


def trial_round(settings: Settings, champion_error: float, challenger_error: float, days: int):
    """Chronos-2 and naive with `days` matured as-of dates, then one round the next day."""
    config = TournamentConfig(families=["chronos"], trial_days=60)
    champion = contender("chronos", "champion", 1, CHRONOS)
    challenger = contender("chronos", "challenger", 2, CHRONOS)
    naive = contender("naive", "baseline", 1, NAIVE)
    history = trading_days(days + 1)
    add_forecasts(settings, champion, history[:-1], champion_error)
    add_forecasts(settings, challenger, history[:-1], challenger_error)
    add_forecasts(settings, naive, history[:-1], 0.6)
    for item in (champion, challenger, naive):
        add_forecasts(settings, item, history[-1:], 0.0, matured=False)
    registry = FakeRegistry([champion, challenger, naive])
    tournament = ModelTournament(config, settings.db_url, registry)
    return tournament, registry, history[-1]


def test_challenger_waits_for_60_days(settings: Settings) -> None:
    tournament, registry, today = trial_round(settings, 0.5, 0.3, days=59)
    outcome = tournament.run(today, "t" * 32, RUN_TIME)
    trial = outcome.trials[0]
    assert (trial.trial_days, trial.required_days) == (59, 60)
    assert trial.challenger_ahead
    assert not trial.promoted
    assert trial.status == "in trials"
    assert outcome.promoted == []
    assert registry.contenders["chronos:champion"].version == "1"


def test_challenger_promoted_after_60_days_ahead(settings: Settings) -> None:
    tournament, registry, today = trial_round(settings, 0.5, 0.3, days=60)
    outcome = tournament.run(today, "t" * 32, RUN_TIME)
    assert outcome.promoted == ["chronos"]
    assert registry.contenders["chronos:champion"].version == "2"
    assert registry.contenders["chronos:challenger"].version == "1"
    trials = read(
        settings,
        "SELECT family, trial_days, required_days, champion_mae, challenger_mae, "
        "challenger_ahead, promoted, promoted_at IS NOT NULL, status FROM challenger_trials",
    )
    assert trials == [
        ("chronos", 60, 60, pytest.approx(0.5), pytest.approx(0.3), 1, 1, 1, "promoted")
    ]
    history = read(
        settings,
        "SELECT family, registered_model, new_champion_version, former_champion_version, "
        "trial_days FROM promotion_history",
    )
    assert history == [("chronos", "natgas-chronos", "2", "1", 60)]
    # The new champion won every event with its record, and its forecasts were published
    winners = read(settings, "SELECT stage, model_version FROM event_winners")
    assert winners == [("chronos:champion", "2")] * 3
    published = read(
        settings, "SELECT DISTINCT model, model_version FROM predictions WHERE stage = 'production'"
    )
    assert published == [("chronos", "2")]

    # A rerun the same day keeps the promotion as recorded and does not promote again
    again = tournament.run(today, "u" * 32, RUN_TIME)
    assert again.promoted == []
    assert read(settings, "SELECT COUNT(*) FROM promotion_history") == [(1,)]
    assert read(settings, "SELECT status FROM challenger_trials") == [("promoted",)]


def test_new_challenger_starts_its_trial_after_the_promotion(settings: Settings) -> None:
    tournament, registry, today = trial_round(settings, 0.5, 0.3, days=60)
    tournament.run(today, "t" * 32, RUN_TIME)
    tomorrow = (pd.Timestamp(today) + pd.offsets.BDay(1)).date()
    for item in registry.load_all():
        add_forecasts(settings, item, [tomorrow], 0.0, matured=False)
    outcome = tournament.run(tomorrow, "v" * 32, RUN_TIME)
    trial = outcome.trials[0]
    assert trial.challenger_version == "1"
    assert trial.trial_started == tomorrow
    assert trial.trial_days == 0


def test_champion_defends_its_title(settings: Settings) -> None:
    tournament, registry, today = trial_round(settings, 0.3, 0.5, days=60)
    outcome = tournament.run(today, "t" * 32, RUN_TIME)
    assert outcome.promoted == []
    assert outcome.trials[0].status == "champion defended"
    assert registry.contenders["chronos:champion"].version == "1"


def test_round_writes_standings_and_winners(settings: Settings) -> None:
    config = TournamentConfig(families=["chronos", "gradient_boosting"])
    today = date(2026, 11, 18)
    naive_backtest = {
        "days 1 to 5": 0.2476,
        "days 6 to 15": 0.4656,
        "days 16 to 30": 0.6589,
        "all days": 0.5259,
    }
    contenders = [
        contender(
            "chronos",
            "champion",
            1,
            CHRONOS,
            {
                "days 1 to 5": 0.2518,
                "days 6 to 15": 0.4628,
                "days 16 to 30": 0.6459,
                "all days": 0.5192,
            },
        ),
        contender(
            "chronos",
            "challenger",
            2,
            CHRONOS,
            {"days 1 to 5": 0.26, "days 6 to 15": 0.47, "days 16 to 30": 0.65, "all days": 0.5225},
        ),
        contender(
            "gradient_boosting",
            "champion",
            1,
            GRADIENT_BOOSTING,
            {
                "days 1 to 5": 0.2473,
                "days 6 to 15": 0.4594,
                "days 16 to 30": 0.6537,
                "all days": 0.5212,
            },
        ),
        contender(
            "gradient_boosting",
            "challenger",
            2,
            GRADIENT_BOOSTING,
            {"days 1 to 5": 0.25, "days 6 to 15": 0.46, "days 16 to 30": 0.66, "all days": 0.5236},
        ),
        contender("naive", "baseline", 1, NAIVE, naive_backtest),
    ]
    for item in contenders:
        add_forecasts(settings, item, [today], 0.0, matured=False)
    tournament = ModelTournament(config, settings.db_url, FakeRegistry(contenders))
    outcome = tournament.run(today, "w" * 32, RUN_TIME)

    # Gradient boosting is 0.12 percent better than naive on days 1 to 5, short of the margin.
    # It clears the margin on days 6 to 15, and Chronos-2 on days 16 to 30.
    assert outcome.winners() == {
        "days 1 to 5": "baseline:naive v1",
        "days 6 to 15": "gradient_boosting:champion v1",
        "days 16 to 30": "chronos:champion v1",
    }
    assert outcome.standings == 4 * 5
    assert outcome.stitched == 30
    standings = read(
        settings,
        "SELECT horizon_bucket, stage, rank, qualified, is_winner, prod_weight, blended_mae "
        "FROM tournament_standings WHERE horizon_bucket = 'days 16 to 30' ORDER BY rank",
    )
    assert [row[1] for row in standings][:2] == ["chronos:champion", "chronos:challenger"]
    by_stage = {row[1]: row for row in standings}
    assert by_stage["chronos:champion"][3:5] == (1, 1)
    assert by_stage["chronos:challenger"][3:5] == (0, 0)
    assert by_stage["baseline:naive"][3:5] == (1, 0)
    assert by_stage["chronos:champion"][5:] == (0.0, pytest.approx(0.6459))
    winners = read(
        settings,
        "SELECT horizon_bucket, first_day, last_day, model, qualified_count FROM event_winners "
        "ORDER BY first_day",
    )
    assert winners == [
        ("days 1 to 5", 1, 5, "naive", 0),
        ("days 6 to 15", 6, 15, "gradient_boosting", 1),
        ("days 16 to 30", 16, 30, "chronos", 1),
    ]
    assert read(settings, "SELECT COUNT(DISTINCT horizon_bucket) FROM tournament_standings") == [
        (4,)
    ]


# Seeding the per family registries from a grid search


@pytest.fixture
def filled_settings(settings: Settings) -> Settings:
    prices = pd.bdate_range(end=pd.Timestamp(LAST_PRICE), periods=500)
    rng = np.random.default_rng(0)
    highs = 3 + np.cumsum(rng.normal(0, 0.05, len(prices)))
    rows = [
        {"date": day.date(), "high": float(high), "source": "test"}
        for day, high in zip(prices, highs, strict=True)
    ]
    weather_days = pd.date_range(prices[0], pd.Timestamp(LAST_PRICE) + pd.Timedelta(days=14))
    hdd = 15 + 15 * np.cos(2 * np.pi * weather_days.dayofyear / 365.25)
    weather = [
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
        upsert(session, Price, rows)
        upsert(session, WeatherDaily, weather)
    return settings


class ContextChronosPipeline(FakeChronosPipeline):
    """Like the shared fake, but longer contexts and weather move the forecast a little,
    so different Chronos-2 configs get different scores."""

    def predict_df(self, df, future_df=None, **kwargs):
        frame = super().predict_df(df, future_df=future_df, **kwargs)
        shift = 0.001 * len(df) / 64 + (
            0.002 if future_df is not None and "hdd" in future_df else 0
        )
        frame["predictions"] += shift
        for column in frame.columns:
            if column.startswith("0."):
                frame[column] += shift
        return frame


@pytest.fixture
def seeded(filled_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A small grid search, then the tournament seeded from it."""
    key = f"{ChronosModelConfig().model_id}@cpu"
    monkeypatch.setattr(ChronosModel, "pipelines", {key: ContextChronosPipeline()})
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("natgas-grid-search")
    provider = ForecastDataProvider.from_settings(filled_settings, 30)
    dates = provider.backtest_dates("2024-01-01", "2024-04-30", step_days=30)
    optimizer = GridSearchOptimizer(Backtester(provider, dates))
    models = {
        "naive": {},
        "gradient_boosting": {"grid": {"max_iter": [20], "n_lags": [3, 5], "use_weather": [False]}},
        "chronos": {"grid": {"context_length": [64, 128], "use_weather": [True, False]}},
    }
    with mlflow.start_run(run_name="grid-search-notebook"):
        mlflow.log_params({"backtest_count": len(dates), "horizon_days": 30})
        optimizer.run(models)
    runs = GridSearchRuns(tracking_uri)
    config = TournamentConfig(families=["chronos", "gradient_boosting"], tracking_uri=tracking_uri)
    seeder = TournamentSeeder(runs, provider, config, pd.Timestamp(LAST_PRICE))
    return seeder, seeder.seed(), config


def test_seeding_gives_every_family_two_valid_titles(seeded) -> None:
    seeder, versions, config = seeded
    assert [(item.registered_model, item.role, item.version) for item in versions] == [
        ("natgas-chronos", "champion", "1"),
        ("natgas-chronos", "challenger", "2"),
        ("natgas-gradient-boosting", "champion", "1"),
        ("natgas-gradient-boosting", "challenger", "2"),
        ("natgas-naive", "baseline", "1"),
    ]
    contenders = {item.stage: item for item in FamilyRegistry(config).load_all()}
    assert set(contenders) == {
        "chronos:champion",
        "chronos:challenger",
        "gradient_boosting:champion",
        "gradient_boosting:challenger",
        "baseline:naive",
    }
    # Chronos-2 must use weather, so the no weather configs are never seeded
    assert contenders["chronos:champion"].config.use_weather
    assert contenders["chronos:challenger"].config.use_weather
    assert (
        contenders["chronos:champion"].config.context_length
        != contenders["chronos:challenger"].config.context_length
    )
    # The champion is the better of the two in the grid search
    by_role = {(item.family, item.role): item.mae for item in versions}
    assert by_role[("chronos", "champion")] <= by_role[("chronos", "challenger")]
    # Backtest MAE of every bucket is on the version for the scorecard
    assert set(contenders["gradient_boosting:champion"].backtest_mae) == {
        "days 1 to 5",
        "days 6 to 15",
        "days 16 to 30",
        "all days",
    }
    assert contenders["baseline:naive"].backtest_mae["all days"] == pytest.approx(
        by_role[("naive", "baseline")], abs=1e-6
    )
    with pytest.raises(ValueError, match="already have a champion"):
        seeder.seed()


def test_promotion_swaps_the_registry_aliases(seeded) -> None:
    _, _, config = seeded
    registry = FamilyRegistry(config)
    assert registry.promote("chronos") == ("2", "1")
    client = MlflowClient(config.tracking_uri)
    aliases = client.get_registered_model("natgas-chronos").aliases
    assert aliases == {"champion": 2, "challenger": 1, "legacy_champion": 1}


def test_pick_skips_rule_breakers_and_configs_that_forecast_the_same(seeded) -> None:
    seeder, _, _ = seeded

    def run(mae, **params):
        values = {name: str(value) for name, value in params.items()}
        return SimpleNamespace(
            data=SimpleNamespace(
                metrics={"mae": mae}, tags={"model_name": "greykite"}, params=values
            )
        )

    # Holiday window changes nothing with holiday dummies off, so the scores tie
    runs = [
        run(0.7362, use_holidays=False, holiday_window=4),
        run(0.7362, use_holidays=False, holiday_window=2),
        run(0.7479, use_holidays=True, holiday_window=2),
    ]
    picked = seeder.pick(runs)
    assert [item.data.params["holiday_window"] for item in picked] == ["4", "2"]
    assert picked[1].data.params["use_holidays"] == "True"


# Database changes


def test_stage_column_migration_keeps_every_row(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = get_engine(db_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE predictions (as_of_date DATE NOT NULL, stage VARCHAR(16) NOT NULL, "
                "horizon_day INTEGER NOT NULL, run_id VARCHAR(32) NOT NULL, "
                "run_timestamp DATETIME NOT NULL, target_date DATE NOT NULL, "
                "model VARCHAR(32) NOT NULL, model_version VARCHAR(64) NOT NULL, "
                "forecast FLOAT NOT NULL, lower FLOAT, upper FLOAT, "
                "PRIMARY KEY (as_of_date, stage, horizon_day))"
            )
        )
        connection.execute(text("CREATE INDEX ix_predictions_run_id ON predictions (run_id)"))
        connection.execute(
            text(
                "INSERT INTO predictions VALUES ('2026-11-18', 'production', 1, 'abc', "
                "'2026-11-19 12:00:00', '2026-11-19', 'chronos', '8', 3.1, 2.9, 3.3)"
            )
        )
    init_db(db_url)
    migrator = StageColumnMigrator(db_url)
    assert migrator.pending() == ["predictions"]
    assert migrator.migrate() == ["predictions"]
    assert migrator.migrate() == []
    columns = {column["name"]: column for column in inspect(engine).get_columns("predictions")}
    assert columns["stage"]["type"].length == 32
    with engine.connect() as connection:
        assert connection.execute(text("SELECT stage, forecast FROM predictions")).all() == [
            ("production", 3.1)
        ]
        views = connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'view'")).all()
    assert {name for (name,) in views} == {"forecast_accuracy", "forecast_accuracy_rolling"}


def test_legacy_rows_are_relabeled_once(settings: Settings) -> None:
    legacy = contender("chronos", "champion", 8, CHRONOS)
    add_forecasts(settings, legacy, [date(2026, 11, 18)], 0.1, stage="production")
    add_forecasts(settings, legacy, [date(2026, 11, 18)], 0.1, stage="staging")
    relabeler = LegacyStageRelabeler(settings.db_url)
    assert relabeler.relabel() == 4 * 30
    assert relabeler.relabel() == 0
    stages = read(settings, "SELECT DISTINCT stage FROM predictions ORDER BY stage")
    assert stages == [("legacy:challenger",), ("legacy:champion",)]


def test_standings_command_before_any_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NATGAS_DB_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    result = CliRunner().invoke(cli.app, ["tournament", "standings"])
    assert result.exit_code == 0
    assert "no tournament results yet" in result.output

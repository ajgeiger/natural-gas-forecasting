from datetime import date, datetime, timezone

import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from natgas.forecasting.production import ScoringFlow, ScoringRun


@pytest.fixture(scope="module", autouse=True)
def prefect_backend():
    with prefect_test_harness():
        yield


class FakePipeline:
    """Records the stage calls the flow makes. refresh_data fails `failures` times first."""

    def __init__(self, refresh: bool = True, failures: int = 0) -> None:
        self.refresh = refresh
        self.failures = failures
        self.calls: list[str] = []

    def refresh_data(self) -> dict[str, int]:
        self.calls.append("refresh_data")
        if self.failures:
            self.failures -= 1
            raise ConnectionError("yahoo is down")
        return {"prices": 21, "weather": 44, "storage": 4} if self.refresh else {}

    def score_models(self) -> ScoringRun:
        self.calls.append("score_models")
        return ScoringRun(
            run_id="abc123",
            run_timestamp=datetime(2026, 11, 19, 12, tzinfo=timezone.utc),
            as_of_date=pd.Timestamp("2026-11-18"),
            loaded={},
            saved={"production": 30, "staging": 30},
            actuals=0,
            weather_days={"forecast": 10, "normal": 20},
        )

    def backfill_actuals(self, as_of_date: date) -> int:
        self.calls.append(f"backfill_actuals {as_of_date}")
        return 14


def use_pipeline(monkeypatch, pipeline: FakePipeline) -> None:
    monkeypatch.setattr(ScoringFlow, "pipeline_for", staticmethod(lambda config, refresh: pipeline))


def test_flow_runs_the_three_tasks_in_order(monkeypatch) -> None:
    pipeline = FakePipeline()
    use_pipeline(monkeypatch, pipeline)
    summary = ScoringFlow().run_once()
    assert pipeline.calls == ["refresh_data", "score_models", "backfill_actuals 2026-11-18"]
    assert summary == {
        "run_id": "abc123",
        "run_timestamp": "2026-11-19T12:00:00+00:00",
        "as_of_date": "2026-11-18",
        "loaded": {"prices": 21, "weather": 44, "storage": 4},
        "saved": {"production": 30, "staging": 30},
        "weather_days": {"forecast": 10, "normal": 20},
        "actuals": 14,
    }


def test_no_refresh_still_reports_an_empty_load(monkeypatch) -> None:
    pipeline = FakePipeline(refresh=False)
    use_pipeline(monkeypatch, pipeline)
    summary = ScoringFlow().run_once(refresh=False)
    assert summary["loaded"] == {}
    assert pipeline.calls[0] == "refresh_data"


def test_refresh_retries_then_succeeds(monkeypatch) -> None:
    fast = ScoringFlow.refresh_data.with_options(retry_delay_seconds=0)
    monkeypatch.setattr(ScoringFlow, "refresh_data", staticmethod(fast))
    pipeline = FakePipeline(failures=2)
    use_pipeline(monkeypatch, pipeline)
    summary = ScoringFlow().run_once()
    assert pipeline.calls[:3] == ["refresh_data"] * 3
    assert summary["actuals"] == 14


def test_refresh_failure_fails_the_flow_and_stops_scoring(monkeypatch) -> None:
    assert ScoringFlow.refresh_data.retries == 2
    assert ScoringFlow.refresh_data.retry_delay_seconds == 30
    fast = ScoringFlow.refresh_data.with_options(retry_delay_seconds=0)
    monkeypatch.setattr(ScoringFlow, "refresh_data", staticmethod(fast))
    pipeline = FakePipeline(failures=5)
    use_pipeline(monkeypatch, pipeline)

    state = ScoringFlow.daily_scoring(return_state=True)
    assert state.is_failed()
    with pytest.raises(ConnectionError, match="yahoo is down"):
        state.result()
    # One try plus two retries, and scoring never ran
    assert pipeline.calls == ["refresh_data"] * 3


def test_serve_builds_the_cron_schedule(monkeypatch) -> None:
    captured = {}

    def fake_serve(self, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(type(ScoringFlow.daily_scoring), "serve", fake_serve)
    ScoringFlow("config/tuning.yaml").serve()

    schedule = captured["schedule"]
    assert captured["name"] == "natgas-daily-scoring"
    assert schedule.cron == "0 7 * * 2-6"
    assert schedule.timezone == "America/New_York"
    assert captured["parameters"] == {"refresh": True, "config": "config/tuning.yaml"}
    assert captured["global_limit"] == 1


def test_deployment_points_at_the_class_flow() -> None:
    deployment = ScoringFlow.daily_scoring.to_deployment(
        name="natgas-daily-scoring", cron="0 7 * * 2-6"
    )
    assert deployment.entrypoint.endswith("ScoringFlow.py:ScoringFlow.daily_scoring")
    assert ScoringFlow.daily_scoring.name == "natgas-daily-scoring"

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from natgas.config import Location, Settings
from natgas.db import init_db
from natgas.forecasting.models import ChronosModel, ChronosModelConfig


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    init_db(db_url)
    return Settings(
        db_url=db_url,
        default_start=date(2015, 1, 1),
        ticker="NG=F",
        storage_series="NW2_EPG0_SWO_R48_BCF",
        storage_region="lower48",
        base_temp_f=65.0,
        forecast_days=3,
        locations=[Location(name="chicago", latitude=41.88, longitude=-87.63)],
        eia_api_key=None,
    )


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
        # Quantiles spread around the median so intervals can be checked
        for level in quantile_levels or [0.5]:
            frame[str(level)] = [value + level - 0.5 for value in values]
        return frame


@pytest.fixture(autouse=True)
def fake_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    key = f"{ChronosModelConfig().model_id}@cpu"
    monkeypatch.setattr(ChronosModel, "pipelines", {key: FakeChronosPipeline()})

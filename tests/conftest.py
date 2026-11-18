from datetime import date
from pathlib import Path

import pytest

from natgas.config import Location, Settings
from natgas.db import init_db


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

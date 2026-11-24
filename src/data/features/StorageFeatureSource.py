"""EIA weekly storage, lined up with the day each report was published."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from data.database.SqlReader import SqlReader
from data.features.AbstractFeatureSource import AbstractFeatureSource

COLUMN = "storage_bcf"

# EIA publishes the week ending Friday on the next Thursday
RELEASE_LAG_DAYS = 6


class StorageFeatureSource(AbstractFeatureSource):
    """Latest published storage level on each date.

    A report only counts once it is out, so a value for the week ending on a
    Friday becomes usable 6 days later. Future dates carry the last report that
    was published on or before the as-of date.
    """

    def __init__(self, weekly_storage: pd.DataFrame) -> None:
        frame = weekly_storage.copy()
        frame["released"] = pd.to_datetime(frame["week_ending"]) + pd.Timedelta(
            days=RELEASE_LAG_DAYS
        )
        self.releases = frame.set_index("released")["value_bcf"].sort_index()

    @classmethod
    def from_database(cls, engine: Engine, region: str) -> StorageFeatureSource:
        query = text("SELECT week_ending, value_bcf FROM storage_weekly WHERE region = :region")
        columns = ["week_ending", "value_bcf"]
        return cls(SqlReader.read(engine, query, columns, {"region": region}))

    @property
    def has_data(self) -> bool:
        return not self.releases.empty

    @property
    def columns(self) -> list[str]:
        return [COLUMN]

    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        known = self.releases.loc[: pd.Timestamp(as_of)]
        # Each date sees the last report out by that date, and never past as_of
        lookup_dates = dates.where(dates <= as_of, pd.Timestamp(as_of))
        positions = known.index.searchsorted(lookup_dates, side="right") - 1
        values = [known.iloc[p] if p >= 0 else float("nan") for p in positions]
        return pd.DataFrame({COLUMN: values}, index=dates)

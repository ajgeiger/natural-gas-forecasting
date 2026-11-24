"""Weekly working gas in storage from the EIA API v2."""

from __future__ import annotations

import logging
from datetime import date

from data.database import StorageWeekly
from de.utils import Loader, get_json, tracked_run

log = logging.getLogger(__name__)

EIA_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
PAGE_SIZE = 5000


class StorageLoader(Loader):
    name = "storage"
    table = StorageWeekly

    def fetch(self, start: date, end: date) -> list[dict]:
        """Raw EIA records for the configured series, following pagination."""
        records: list[dict] = []
        offset = 0
        while True:
            params = {
                "api_key": self.settings.eia_api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": self.settings.storage_series,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": PAGE_SIZE,
            }
            body = get_json(EIA_URL, params)["response"]
            page = body.get("data", [])
            records.extend(page)

            offset += len(page)
            if not page or offset >= int(body.get("total", 0)):
                return records

    def to_rows(self, records: list[dict]) -> list[dict]:
        return [
            {
                "week_ending": date.fromisoformat(record["period"]),
                "region": self.settings.storage_region,
                "value_bcf": float(record["value"]),
            }
            for record in records
            if record.get("value") is not None
        ]

    def rows(self, start: date, end: date) -> list[dict]:
        return self.to_rows(self.fetch(start, end))

    def run(self, start: date, end: date) -> int:
        if not self.settings.eia_api_key:
            message = "EIA_API_KEY is not set, skipping storage"
            log.warning(message)
            with tracked_run(self.settings.db_url, self.name) as run:
                run.skip(message)
            return 0
        return super().run(start, end)

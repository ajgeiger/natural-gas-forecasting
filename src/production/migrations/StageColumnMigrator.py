"""Widens the stage column of existing databases for the tournament stage names."""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from data.database import STAGE_LENGTH, ForecastActual, Prediction, get_engine
from production.publishing.ForecastMonitor import ForecastMonitor

log = logging.getLogger(__name__)

VIEWS = ["forecast_accuracy", "forecast_accuracy_rolling"]


class StageColumnMigrator:
    """Brings predictions and forecast_actuals up to a 32 character stage column.

    Stages such as "gradient_boosting:challenger" need more than the original
    16 characters. SQLite cannot change a column type, so each table is
    renamed, created again from the current model, refilled, and the old copy
    dropped. Postgres alters the column in place. The accuracy views read
    these tables, so they are dropped first and created again after. Nothing
    happens when the columns are already wide enough.
    """

    tables = [Prediction, ForecastActual]

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = get_engine(db_url)

    def stage_length(self, table_name: str) -> int | None:
        inspector = inspect(self.engine)
        if not inspector.has_table(table_name):
            return None
        for column in inspector.get_columns(table_name):
            if column["name"] == "stage":
                return getattr(column["type"], "length", None)
        return None

    def pending(self) -> list[str]:
        names = [model.__tablename__ for model in self.tables]
        return [
            name
            for name in names
            if (length := self.stage_length(name)) is not None and length < STAGE_LENGTH
        ]

    def migrate(self) -> list[str]:
        """Names of the tables that were migrated."""
        pending = self.pending()
        if not pending:
            return []
        models = {model.__tablename__: model for model in self.tables}
        sqlite = self.engine.dialect.name == "sqlite"
        with self.engine.begin() as connection:
            for view in VIEWS:
                connection.execute(text(f"DROP VIEW IF EXISTS {view}"))
            for name in pending:
                table = models[name].__table__
                if not sqlite:
                    connection.execute(
                        text(f"ALTER TABLE {name} ALTER COLUMN stage TYPE VARCHAR({STAGE_LENGTH})")
                    )
                    continue
                columns = ", ".join(column.name for column in table.columns)
                for index in table.indexes:
                    connection.execute(text(f"DROP INDEX IF EXISTS {index.name}"))
                connection.execute(text(f"ALTER TABLE {name} RENAME TO {name}_old"))
                table.create(connection)
                connection.execute(
                    text(f"INSERT INTO {name} ({columns}) SELECT {columns} FROM {name}_old")
                )
                connection.execute(text(f"DROP TABLE {name}_old"))
                log.info("widened %s.stage to %d characters", name, STAGE_LENGTH)
        ForecastMonitor(self.db_url).create_views()
        return pending

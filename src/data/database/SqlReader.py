"""Runs a SQL query and returns a DataFrame."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, TextClause


class SqlReader:
    """Reads query results through SQLAlchemy directly.

    pd.read_sql has broken with some pandas and SQLAlchemy version pairs, so
    this keeps database reads working no matter which pandas is installed.
    """

    @staticmethod
    def read(
        engine: Engine, query: TextClause, columns: list[str], params: dict | None = None
    ) -> pd.DataFrame:
        with engine.connect() as connection:
            rows = connection.execute(query, params or {}).all()
        return pd.DataFrame(rows, columns=columns)

"""Read and write the project database from Spark over JDBC.

The SQL database stays the system of record and the loaders keep writing to
it with upserts. Spark reads the same tables and can write new ones, so jobs
can move to Spark later without moving the data. Needs the spark extra and
Java 17.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from data.database import get_engine

SQLITE_PACKAGE = "org.xerial:sqlite-jdbc:3.46.1.3"
POSTGRES_PACKAGE = "org.postgresql:postgresql:42.7.4"


def jdbc_options(db_url: str) -> dict[str, str]:
    """Spark JDBC options for a SQLAlchemy database url."""
    url = make_url(db_url)
    backend = url.get_backend_name()
    if backend == "sqlite":
        path = Path(url.database).as_posix()
        # date_class=TEXT stores dates written by Spark as text, like SQLAlchemy does
        return {"url": f"jdbc:sqlite:{path}?date_class=TEXT", "driver": "org.sqlite.JDBC"}
    if backend == "postgresql":
        options = {
            "url": f"jdbc:postgresql://{url.host or 'localhost'}:{url.port or 5432}/{url.database}",
            "driver": "org.postgresql.Driver",
        }
        if url.username:
            options["user"] = url.username
        if url.password:
            options["password"] = url.password
        return options
    raise NotImplementedError(f"no Spark driver set up for {backend}")


class SparkDatabase:
    """A local SparkSession pointed at the project database."""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.options = jdbc_options(db_url)
        self.is_sqlite = make_url(db_url).get_backend_name() == "sqlite"
        self._spark = None

    @property
    def spark(self):
        """Started on first use, since starting Spark takes a few seconds."""
        if self._spark is None:
            from pyspark.sql import SparkSession

            # Use this interpreter for Spark's Python workers (matters on Windows)
            os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
            self._spark = (
                SparkSession.builder.master("local[*]")
                .appName("natgas")
                .config(
                    "spark.jars.packages", SQLITE_PACKAGE if self.is_sqlite else POSTGRES_PACKAGE
                )
                # The database stores UTC times, so keep Spark in UTC too
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
                .config("spark.ui.showConsoleProgress", "false")
                .getOrCreate()
            )
        return self._spark

    def read_table(self, name: str):
        """Whole table as a Spark DataFrame."""
        reader = self.spark.read.format("jdbc").options(**self.options).option("dbtable", name)
        fixes = self.sqlite_column_types(name)
        if not fixes:
            return reader.load()

        # Read dates as text and floats as doubles, then cast dates in Spark
        read_types = {
            column: "DOUBLE" if kind == "double" else "STRING" for column, kind in fixes.items()
        }
        schema = ", ".join(f"{column} {kind}" for column, kind in read_types.items())
        frame = reader.option("customSchema", schema).load()
        for column, kind in fixes.items():
            if kind != "double":
                frame = frame.withColumn(column, frame[column].cast(kind))
        return frame

    def write_table(self, frame, name: str, mode: str = "append") -> None:
        """Write a DataFrame. mode is append or overwrite (overwrite recreates the table)."""
        frame.write.format("jdbc").options(**self.options).option("dbtable", name).mode(mode).save()

    def sqlite_column_types(self, name: str) -> dict[str, str]:
        """Spark types for SQLite columns that JDBC reads wrong.

        SQLite keeps dates as text, and the JDBC driver reports FLOAT as 32 bit.
        """
        if not self.is_sqlite:
            return {}
        types = {}
        for column in inspect(get_engine(self.db_url)).get_columns(name):
            type_name = str(column["type"]).upper()
            if type_name == "DATE":
                types[column["name"]] = "date"
            elif "TIME" in type_name:
                types[column["name"]] = "timestamp"
            elif type_name in ("FLOAT", "REAL", "DOUBLE", "DOUBLE PRECISION"):
                types[column["name"]] = "double"
        return types

    def stop(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

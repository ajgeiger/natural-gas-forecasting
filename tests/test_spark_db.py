import os
import shutil
from datetime import date

import pytest

from data.database import Price, session_scope, upsert
from de.spark_db import SparkDatabase, jdbc_options

has_java = bool(shutil.which("java") or os.getenv("JAVA_HOME"))


def test_jdbc_options_sqlite(tmp_path):
    options = jdbc_options(f"sqlite:///{tmp_path / 'test.db'}")
    assert options["url"].startswith("jdbc:sqlite:")
    assert options["url"].endswith("test.db?date_class=TEXT")
    assert options["driver"] == "org.sqlite.JDBC"


def test_jdbc_options_postgres():
    options = jdbc_options("postgresql+psycopg://alex:pw@db.local:5433/natgas")
    assert options["url"] == "jdbc:postgresql://db.local:5433/natgas"
    assert options["user"] == "alex"
    assert options["password"] == "pw"


@pytest.mark.skipif(not has_java, reason="Spark needs Java")
def test_read_and_write_round_trip(settings):
    pytest.importorskip("pyspark")
    row = {"date": date(2024, 1, 2), "high": 2.6, "source": "yahoo"}
    with session_scope(settings.db_url) as session:
        upsert(session, Price, [row])

    database = SparkDatabase(settings.db_url)
    try:
        prices = database.read_table("prices")
        assert prices.collect()[0]["date"] == date(2024, 1, 2)

        database.write_table(prices.select("date", "high"), "prices_copy", mode="overwrite")
        copy = database.read_table("prices_copy").collect()
        assert [(r["date"], r["high"]) for r in copy] == [(date(2024, 1, 2), 2.6)]
    finally:
        database.stop()

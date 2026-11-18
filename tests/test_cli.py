from typer.testing import CliRunner

from natgas import cli

runner = CliRunner()


def test_init_db_creates_file(tmp_path, monkeypatch):
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("NATGAS_DB_URL", f"sqlite:///{db_path}")
    result = runner.invoke(cli.app, ["init-db"])
    assert result.exit_code == 0
    assert db_path.exists()


def test_ingest_all_keeps_going_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("NATGAS_DB_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    called = []

    def broken(*_args):
        called.append("prices")
        raise RuntimeError("no data")

    def working(name):
        def loader(*_args):
            called.append(name)
            return 5

        return loader

    monkeypatch.setattr(cli, "ingest_prices", broken)
    monkeypatch.setattr(cli, "ingest_weather", working("weather"))
    monkeypatch.setattr(cli, "ingest_storage", working("storage"))

    result = runner.invoke(
        cli.app, ["ingest", "all", "--start", "2024-01-01", "--end", "2024-01-31"]
    )
    assert called == ["prices", "weather", "storage"]
    assert result.exit_code == 1


def test_rejects_start_after_end(tmp_path, monkeypatch):
    monkeypatch.setenv("NATGAS_DB_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    result = runner.invoke(
        cli.app, ["ingest", "prices", "--start", "2024-02-01", "--end", "2024-01-01"]
    )
    assert result.exit_code != 0

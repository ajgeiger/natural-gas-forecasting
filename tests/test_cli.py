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

    def fake_loader(name, fails=False):
        class FakeLoader:
            def __init__(self, _settings):
                pass

            def run(self, _start, _end):
                called.append(name)
                if fails:
                    raise RuntimeError("no data")
                return 5

        return FakeLoader

    monkeypatch.setattr(
        cli,
        "LOADERS",
        {
            "prices": fake_loader("prices", fails=True),
            "weather": fake_loader("weather"),
            "storage": fake_loader("storage"),
        },
    )

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

from pathlib import Path

from natgas.config import load_settings, resolve_db_url

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def test_load_settings_from_repo_config(monkeypatch):
    monkeypatch.delenv("NATGAS_DB_URL", raising=False)
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    settings = load_settings(REPO_CONFIG)

    assert settings.ticker == "NG=F"
    assert settings.db_url.endswith("data/natgas.db")
    assert settings.eia_api_key is None
    assert len(settings.locations) == 8


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NATGAS_DB_URL", "postgresql+psycopg://user:pw@localhost/natgas")
    monkeypatch.setenv("EIA_API_KEY", "test-key")
    settings = load_settings(REPO_CONFIG)

    assert settings.db_url == "postgresql+psycopg://user:pw@localhost/natgas"
    assert settings.eia_api_key == "test-key"


def test_relative_sqlite_path_is_resolved(tmp_path):
    url = resolve_db_url("sqlite:///data/test.db", tmp_path)
    assert url == f"sqlite:///{tmp_path / 'data' / 'test.db'}"
    assert (tmp_path / "data").is_dir()

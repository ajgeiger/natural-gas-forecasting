"""Load settings from config/settings.yaml with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Settings:
    db_url: str
    default_start: date
    ticker: str
    storage_series: str
    storage_region: str
    base_temp_f: float
    forecast_days: int
    locations: list[Location] = field(default_factory=list)
    eia_api_key: str | None = None


def find_config_file() -> Path:
    """Use NATGAS_CONFIG if set, then ./config, then the repo the package was installed from."""
    env_path = os.getenv("NATGAS_CONFIG")
    if env_path:
        return Path(env_path)
    for candidate in (
        Path.cwd() / "config" / "settings.yaml",
        PACKAGE_ROOT / "config" / "settings.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config/settings.yaml not found; set NATGAS_CONFIG")


def resolve_db_url(url: str, project_root: Path) -> str:
    """Make relative sqlite paths absolute so the CLI works from any directory."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and url != "sqlite:///:memory:":
        db_path = Path(url[len(prefix) :])
        if not db_path.is_absolute():
            db_path = project_root / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{db_path}"
    return url


def load_settings(config_path: Path | None = None) -> Settings:
    config_path = Path(config_path) if config_path else find_config_file()
    raw = yaml.safe_load(config_path.read_text())
    project_root = config_path.resolve().parent.parent

    db_url = os.getenv("NATGAS_DB_URL") or raw["database"]["url"]
    weather = raw["weather"]

    return Settings(
        db_url=resolve_db_url(db_url, project_root),
        default_start=date.fromisoformat(str(raw["ingest"]["default_start"])),
        ticker=raw["prices"]["ticker"],
        storage_series=raw["storage"]["series"],
        storage_region=raw["storage"]["region"],
        base_temp_f=float(weather["base_temp_f"]),
        forecast_days=int(weather["forecast_days"]),
        locations=[Location(**loc) for loc in weather["locations"]],
        eia_api_key=os.getenv("EIA_API_KEY") or None,
    )

"""Command line entry point: natgas init-db, natgas ingest ..., natgas spark-show."""

from __future__ import annotations

import logging
from datetime import date

import typer

from natgas.config import Settings, load_settings
from natgas.db import init_db
from natgas.de.prices import PriceLoader
from natgas.de.storage import StorageLoader
from natgas.de.weather import WeatherLoader
from natgas.forecasting.tuning import TuningConfig, TuningPipeline

app = typer.Typer(help="Natural gas forecasting pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Load source data into the database.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")

LOADERS = {"prices": PriceLoader, "weather": WeatherLoader, "storage": StorageLoader}

StartOption = typer.Option(None, help="First date, YYYY-MM-DD. Defaults to the config start.")
EndOption = typer.Option(None, help="Last date, YYYY-MM-DD. Defaults to today.")


def prepare(start: str | None, end: str | None) -> tuple[Settings, date, date]:
    """Load settings, make sure tables exist, and parse the date range."""
    settings = load_settings()
    init_db(settings.db_url)
    start_date = date.fromisoformat(start) if start else settings.default_start
    end_date = date.fromisoformat(end) if end else date.today()
    if start_date > end_date:
        raise typer.BadParameter(f"start {start_date} is after end {end_date}")
    return settings, start_date, end_date


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # urllib3 debug lines include full query strings, which can hold the EIA key
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@app.command("init-db")
def init_db_command() -> None:
    """Create the database tables."""
    settings = load_settings()
    init_db(settings.db_url)
    typer.echo(f"database ready: {settings.db_url}")


@ingest_app.command("prices")
def ingest_prices_command(start: str = StartOption, end: str = EndOption) -> None:
    """Daily NG=F prices from Yahoo Finance."""
    settings, start_date, end_date = prepare(start, end)
    rows = LOADERS["prices"](settings).run(start_date, end_date)
    typer.echo(f"prices: {rows} rows")


@ingest_app.command("weather")
def ingest_weather_command(start: str = StartOption, end: str = EndOption) -> None:
    """Daily temperatures and degree days from Open-Meteo."""
    settings, start_date, end_date = prepare(start, end)
    rows = LOADERS["weather"](settings).run(start_date, end_date)
    typer.echo(f"weather: {rows} rows")


@ingest_app.command("storage")
def ingest_storage_command(start: str = StartOption, end: str = EndOption) -> None:
    """Weekly Lower 48 storage from EIA. Needs EIA_API_KEY."""
    settings, start_date, end_date = prepare(start, end)
    rows = LOADERS["storage"](settings).run(start_date, end_date)
    typer.echo(f"storage: {rows} rows")


@ingest_app.command("all")
def ingest_all_command(start: str = StartOption, end: str = EndOption) -> None:
    """Run every loader. A failure in one source does not stop the others."""
    settings, start_date, end_date = prepare(start, end)

    failed = []
    for name, loader_class in LOADERS.items():
        try:
            rows = loader_class(settings).run(start_date, end_date)
            typer.echo(f"{name}: {rows} rows")
        except Exception:
            logging.getLogger(__name__).exception("%s ingest failed", name)
            failed.append(name)

    if failed:
        raise typer.Exit(code=1)


@app.command("tune")
def tune_command(
    config: str = typer.Option("config/tuning.yaml", help="Tuning config file."),
    publish: bool = typer.Option(True, help="Register the best model in MLflow."),
) -> None:
    """Backtest every config, log to MLflow, and publish the best model."""
    settings = load_settings()
    pipeline = TuningPipeline.from_settings(TuningConfig.from_yaml(config), settings)
    outcome = pipeline.run(publish=publish)

    columns = ["model_name", "hyperparameters", "mae", "rmse", "mape", "wape", "bias"]
    if "skill_vs_naive" in outcome.summary:
        columns.append("skill_vs_naive")
    typer.echo(outcome.summary[columns].head(10).to_string(index=False))
    typer.echo(f"best: {outcome.best_model_name} {outcome.best_hyperparameters}")
    if outcome.registered_version:
        typer.echo(
            f"registered {outcome.registered_model_name} version {outcome.registered_version}"
        )


@app.command("spark-show")
def spark_show_command(table: str, rows: int = typer.Option(5, help="Rows to print.")) -> None:
    """Read a table with Spark and print its schema and newest rows. Needs the spark extra."""
    from natgas.de.spark_db import SparkDatabase

    settings = load_settings()
    database = SparkDatabase(settings.db_url)
    try:
        frame = database.read_table(table)
        frame.printSchema()
        frame.orderBy(frame.columns[0], ascending=False).show(rows, truncate=False)
        typer.echo(f"{table}: {frame.count()} rows")
    finally:
        database.stop()


if __name__ == "__main__":
    app()

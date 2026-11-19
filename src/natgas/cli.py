"""Command line entry point: natgas init-db, ingest, tune, score, flow, serve, export-backtests,
spark-show."""

from __future__ import annotations

import logging
from datetime import date

import typer

from natgas.config import Settings, load_settings
from natgas.db import init_db
from natgas.de.prices import PriceLoader
from natgas.de.storage import StorageLoader
from natgas.de.weather import WeatherLoader
from natgas.forecasting.data import ForecastDataProvider
from natgas.forecasting.production import (
    BacktestExporter,
    ChallengerPublisher,
    GridSearchRuns,
    ScoringFlow,
    ScoringPipeline,
)
from natgas.forecasting.tuning import TuningConfig, TuningPipeline

app = typer.Typer(help="Natural gas forecasting pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Load source data into the database.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")

LOADERS = {"prices": PriceLoader, "weather": WeatherLoader, "storage": StorageLoader}

StartOption = typer.Option(None, help="First date, YYYY-MM-DD. Defaults to the config start.")
EndOption = typer.Option(None, help="Last date, YYYY-MM-DD. Defaults to today.")
TuningOption = typer.Option("config/tuning.yaml", help="Tuning config file.")


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
    # Prefect logs every API call through httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
    optimizer: str = typer.Option(None, help="grid or bayesian. Overrides the config."),
    publish: bool = typer.Option(True, help="Register the chosen model in MLflow."),
) -> None:
    """Search configs with backtests, log to MLflow, and publish the winner."""
    tuning = TuningConfig.from_yaml(config)
    if optimizer:
        tuning = tuning.with_values(optimizer=optimizer)
    outcome = TuningPipeline.from_settings(tuning, load_settings()).run(publish=publish)

    columns = ["model_name", "config", "mae", "rmse", "mape", "wape", "bias"]
    if "skill_vs_naive" in outcome.summary:
        columns.append("skill_vs_naive")
    typer.echo(outcome.summary[columns].head(15).to_string(index=False))
    typer.echo(f"decision: {outcome.decision.reason}")
    if outcome.registered_version:
        typer.echo(
            f"registered {outcome.registered_model_name} version {outcome.registered_version}"
        )


def grid_search_setup(config: str) -> tuple[Settings, TuningConfig, GridSearchRuns]:
    """Settings, tuning config, and the grid search results the commands below read."""
    settings = load_settings()
    init_db(settings.db_url)
    tuning = TuningConfig.from_yaml(config)
    runs = GridSearchRuns(
        tuning.tracking_uri,
        horizon_days=tuning.horizon_days,
        selection_metric=tuning.selection_metric,
    )
    return settings, tuning, runs


@app.command("export-backtests")
def export_backtests_command(
    config: str = TuningOption,
    output: str = typer.Option("outputs/backtest_forecasts.parquet", help="Parquet file."),
) -> None:
    """Write the best backtest per model from the latest grid search to the database and Parquet."""
    settings, tuning, runs = grid_search_setup(config)
    provider = ForecastDataProvider.from_settings(
        settings, tuning.horizon_days, tuning.target_column
    )
    exporter = BacktestExporter(
        runs, provider, settings.db_url, tuning.registered_model_name, tuning.n_jobs
    )
    frame = exporter.build()
    rows = exporter.write(frame, output)
    typer.echo(frame.groupby(["model", "model_version"]).size().to_string())
    typer.echo(f"backtest_forecasts: {rows} rows, also in {output}")


@app.command("register-challenger")
def register_challenger_command(
    config: str = TuningOption,
    model: str = typer.Option("gradient_boosting", help="Model whose best config to register."),
) -> None:
    """Register the best grid search config of a model under the challenger alias."""
    settings, tuning, runs = grid_search_setup(config)
    provider = ForecastDataProvider.from_settings(
        settings, tuning.horizon_days, tuning.target_column
    )
    publisher = ChallengerPublisher(
        runs, provider, tuning.registered_model_name, model, n_jobs=tuning.n_jobs
    )
    version = publisher.publish()
    typer.echo(f"registered {tuning.registered_model_name} version {version} as @challenger")


@app.command("score")
def score_command(
    config: str = TuningOption,
    refresh: bool = typer.Option(True, help="Load the newest data first."),
) -> None:
    """Refresh data, score production and staging, and backfill actuals. Safe to rerun."""
    settings = load_settings()
    run = ScoringPipeline(settings, TuningConfig.from_yaml(config), refresh=refresh).run()
    typer.echo(f"run {run.run_id} as of {run.as_of_date.date()}")
    typer.echo(f"loaded: {run.loaded}")
    typer.echo(f"predictions saved: {run.saved}")
    typer.echo(f"future weather days: {run.weather_days}")
    typer.echo(f"matured forecasts in forecast_actuals: {run.actuals}")


@app.command("flow")
def flow_command(
    config: str = TuningOption,
    refresh: bool = typer.Option(True, help="Load the newest data first."),
) -> None:
    """Run the scoring job once as a Prefect flow, tracked in the Prefect UI. Safe to rerun."""
    summary = ScoringFlow(config).run_once(refresh=refresh)
    typer.echo(f"run {summary['run_id']} as of {summary['as_of_date']}")
    typer.echo(f"loaded: {summary['loaded']}")
    typer.echo(f"predictions saved: {summary['saved']}")
    typer.echo(f"future weather days: {summary['weather_days']}")
    typer.echo(f"matured forecasts in forecast_actuals: {summary['actuals']}")


@app.command("serve")
def serve_command(
    config: str = TuningOption,
    cron: str = typer.Option("0 7 * * 2-6", help="Cron schedule. Default Tuesday to Saturday."),
    timezone: str = typer.Option("America/New_York", help="Timezone of the cron schedule."),
) -> None:
    """Serve the scheduled natgas-daily-scoring deployment until stopped with Ctrl+C.

    Run from the repo root with a Prefect server up and PREFECT_API_URL pointing at it.
    """
    ScoringFlow(config).serve(cron=cron, timezone=timezone)


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

"""Command line entry point: natgas init-db, ingest, tune, tournament, score, flow, serve,
export-backtests, spark-show."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

# Torch must load before numpy on Windows (WinError 1114)
import torch  # noqa: F401
import typer

from config import Settings, load_settings
from data import ForecastDataProvider
from data.database import init_db
from de.prices import PriceLoader
from de.storage import StorageLoader
from de.weather import WeatherLoader
from optimization.config.TuningConfig import TuningConfig
from optimization.pipeline.TuningPipeline import TuningPipeline
from production.migrations.LegacyStageRelabeler import LegacyStageRelabeler
from production.migrations.StageColumnMigrator import StageColumnMigrator
from production.publishing.BacktestExporter import BacktestExporter
from production.registry.GridSearchRuns import GridSearchRuns
from production.scoring.ScoringFlow import ScoringFlow
from production.scoring.ScoringPipeline import MARKET_TIMEZONE, ScoringPipeline
from production.simulation.BacktestTagRefresher import BacktestTagRefresher
from production.simulation.BacktestTrimmer import BacktestTrimmer
from production.simulation.ProductionHistoryReset import ProductionHistoryReset
from production.simulation.TournamentSimulator import TournamentSimulator
from production.tournament.TournamentConfig import TournamentConfig
from production.tournament.TournamentSeeder import TournamentSeeder
from production.tournament.TournamentStore import TournamentStore

app = typer.Typer(help="Natural gas forecasting pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Load source data into the database.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
tournament_app = typer.Typer(
    help="Family trials, horizon events, and the published forecast.", no_args_is_help=True
)
app.add_typer(tournament_app, name="tournament")

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


@tournament_app.command("seed")
def tournament_seed_command(
    config: str = TuningOption,
    force: bool = typer.Option(False, help="Seed again even if a family has a champion."),
    relabel_legacy: bool = typer.Option(
        True, help="Rename pre-tournament production and staging rows to legacy stages."
    ),
) -> None:
    """Give each family a champion and a challenger from the latest grid search."""
    settings, tuning, runs = grid_search_setup(config)
    tournament = TournamentConfig.from_yaml(config)
    provider = ForecastDataProvider.from_settings(
        settings, tuning.horizon_days, tuning.target_column
    )
    # Train through the last complete trading day, the same as-of date scoring uses
    today = datetime.now(MARKET_TIMEZONE).date()
    as_of = ScoringPipeline.last_complete_day(provider, today)
    seeded = TournamentSeeder(runs, provider, tournament, as_of).seed(force=force)
    for item in seeded:
        typer.echo(
            f"{item.registered_model}@{item.role}: version {item.version} "
            f"({item.source_run}, mae {item.mae:.4f})"
        )
    StageColumnMigrator(settings.db_url).migrate()
    if relabel_legacy:
        renamed = LegacyStageRelabeler(settings.db_url).relabel()
        typer.echo(f"legacy rows relabeled: {renamed}")


@tournament_app.command("standings")
def tournament_standings_command() -> None:
    """Print the latest event winners, trials, and standings."""
    settings = load_settings()
    init_db(settings.db_url)
    store = TournamentStore(settings.db_url)
    winners = store.latest("event_winners")
    if winners.empty:
        typer.echo("no tournament results yet, run natgas score")
        return
    columns = ["horizon_bucket", "stage", "model_version", "blended_mae", "improvement"]
    typer.echo(f"event winners as of {winners['as_of_date'].iloc[0]}:")
    typer.echo(winners[columns].to_string(index=False))
    trials = store.latest("challenger_trials")
    columns = ["family", "trial_days", "required_days", "champion_mae", "challenger_mae", "status"]
    typer.echo("\nchallenger trials:")
    typer.echo(trials[columns].to_string(index=False))
    standings = store.latest("tournament_standings")
    columns = ["horizon_bucket", "rank", "stage", "prod_days", "blended_mae", "qualified"]
    typer.echo("\nstandings:")
    typer.echo(standings.sort_values(["horizon_bucket", "rank"])[columns].to_string(index=False))


@tournament_app.command("simulate")
def tournament_simulate_command(
    config: str = TuningOption,
    end: str = typer.Option(None, help="Last day to replay, YYYY-MM-DD. Defaults to yesterday."),
    days: int = typer.Option(60, help="Trading days to replay, ending on end."),
    trim_backtests: bool = typer.Option(
        False, help="Drop backtests from the replayed days and rescore the backtest tags."
    ),
    reset: bool = typer.Option(False, help="Empty the production and tournament tables first."),
    parquet: str = typer.Option("outputs/backtest_forecasts.parquet", help="Backtest Parquet."),
) -> None:
    """Replay past trading days as production, one day at a time, with no look-ahead."""
    settings = load_settings()
    init_db(settings.db_url)
    tuning = TuningConfig.from_yaml(config)
    tournament = TournamentConfig.from_yaml(config)
    last_day = (
        date.fromisoformat(end) if end else datetime.now(MARKET_TIMEZONE).date() - timedelta(days=1)
    )
    simulator = TournamentSimulator(settings, tournament, tuning.horizon_days, tuning.target_column)
    sim_days = simulator.simulation_days(last_day, days)
    first_day = sim_days[0].date()
    typer.echo(f"replaying {len(sim_days)} trading days, {first_day} to {sim_days[-1].date()}")
    if trim_backtests:
        trimmed = BacktestTrimmer(settings.db_url, parquet).trim(first_day)
        typer.echo(f"backtests on or after {first_day} removed: {trimmed}")
        for item in BacktestTagRefresher(tournament).refresh(first_day):
            mae = item["metrics"].get("mae")
            score = f"mae {mae:.4f}" if mae is not None else "tags removed"
            typer.echo(f"{item['name']} v{item['version']}: {score} from {item['rows']} rows")
    if reset:
        typer.echo(f"cleared: {ProductionHistoryReset(settings.db_url).clear()}")
    for run in simulator.run(last_day, days):
        winners = ", ".join(f"{bucket}: {stage}" for bucket, stage in run.events.items())
        promoted = ", ".join(run.promoted) or "none"
        typer.echo(
            f"{run.as_of_date.date()} {run.run_id}: {run.actuals} matured, "
            f"{winners}, promoted {promoted}"
        )


def echo_run(summary: dict) -> None:
    typer.echo(f"run {summary['run_id']} as of {summary['as_of_date']}")
    typer.echo(f"loaded: {summary['loaded']}")
    typer.echo(f"predictions saved: {summary['saved']}")
    typer.echo(f"future weather days: {summary['weather_days']}")
    typer.echo(f"matured forecasts in forecast_actuals: {summary['actuals']}")
    for bucket, winner in summary["events"].items():
        typer.echo(f"{bucket} winner: {winner}")
    typer.echo(f"promoted: {summary['promoted'] or 'none'}")
    typer.echo(f"production rows published: {summary['stitched']}")


@app.command("score")
def score_command(
    config: str = TuningOption,
    refresh: bool = typer.Option(True, help="Load the newest data first."),
) -> None:
    """Refresh data, score every contender, backfill actuals, and run the tournament."""
    run = ScoringPipeline(
        load_settings(),
        TuningConfig.from_yaml(config),
        refresh=refresh,
        tournament=TournamentConfig.from_yaml(config),
    ).run()
    echo_run(ScoringFlow.summary(run))


@app.command("flow")
def flow_command(
    config: str = TuningOption,
    refresh: bool = typer.Option(True, help="Load the newest data first."),
) -> None:
    """Run the scoring job once as a Prefect flow, tracked in the Prefect UI. Safe to rerun."""
    echo_run(ScoringFlow(config).run_once(refresh=refresh))


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
    from de.spark_db import SparkDatabase

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

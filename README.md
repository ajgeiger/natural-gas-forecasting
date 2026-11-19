# natural-gas-forecasting

**Start here: [grid search backtest notebook](experiments/grid_search_backtest.ipynb).** It runs every model on the same 125 rolling as-of dates, forecasts 30 trading days ahead from each, scores them, and publishes the winner to the MLflow model registry. The [Bayesian search notebook](experiments/bayesian_backtest.ipynb) repeats the search with Optuna.

Daily forecasts of the natural gas front-month futures price (Yahoo Finance `NG=F`, daily high), using weather in major gas-demand cities and EIA weekly storage as inputs.

## Why it matters

Natural gas is one of the most volatile commodities there is. Utilities, power plants, and industrial buyers plan purchases days and weeks ahead, and a few cents per MMBtu on millions of MMBtu is real money. Weather drives demand, storage shows how tight supply is, and the futures price reacts to both.

Most forecasting demos stop at one model on one train and test split. This project is built the way a production team would want it: every model is backtested on the same dates with no look-ahead, every run is logged, and a model only replaces the baseline when it earns it.

## What it shows

Prices from 2015-01-02 to 2026-11-18. 125 as-of dates every 14 days from 2021-12-31 to 2026-10-02, each forecasting the next 30 trading days (`backtest.horizon_days` in `config/tuning.yaml`). The two as-of dates after 2026-10-02 are skipped because 30 trading days of actuals do not exist yet. Same dates for every model. Best config per model, MAE, RMSE, and bias in dollars per MMBtu, MAPE and WAPE in percent.

Grid search (61 configs):

| Model | MAE | RMSE | MAPE | WAPE | Bias | Best config |
| --- | --- | --- | --- | --- | --- | --- |
| Chronos-2 | 0.5192 | 0.7975 | 13.12 | 13.65 | -0.0420 | context 512, weather, yearly off |
| Gradient boosting | 0.5212 | 0.8028 | 13.41 | 13.70 | 0.0077 | 5 lags, depth 2, learning rate 0.03, no weather, yearly on |
| Naive | 0.5259 | 0.8107 | 13.50 | 13.82 | 0.0037 | none |
| Greykite | 0.7362 | 1.0741 | 19.36 | 19.35 | 0.0030 | 4 years, 3 price lags, holiday dummies off |

Bayesian search (25 configs, 8 trials per model):

| Model | MAE | RMSE | MAPE | WAPE | Bias | Best config |
| --- | --- | --- | --- | --- | --- | --- |
| Chronos-2 | 0.5221 | 0.7893 | 13.37 | 13.72 | -0.0273 | context 320, no weather, yearly off |
| Naive | 0.5259 | 0.8107 | 13.50 | 13.82 | 0.0037 | none |
| Gradient boosting | 0.5343 | 0.8203 | 13.72 | 14.04 | -0.0010 | 4 lags, depth 2, learning rate 0.019, min leaf 20, weather, yearly on |
| Greykite | 0.7821 | 1.1131 | 20.68 | 20.56 | -0.0172 | 6.9 years, 1 price lag, ridge, holiday dummies off |

MAE by horizon bucket, best grid config per model:

| Model | Days 1 to 5 | Days 6 to 15 | Days 16 to 30 |
| --- | --- | --- | --- |
| Chronos-2 | 0.2518 | 0.4628 | 0.6459 |
| Gradient boosting | 0.2473 | 0.4594 | 0.6537 |
| Naive | 0.2476 | 0.4656 | 0.6589 |
| Greykite | 0.3039 | 0.6123 | 0.9629 |

Yearly calendar features (2 Fourier harmonics of the day of year) are the only calendar option tuned. Weekly harmonics and the after weekend flag stay off. Best config with yearly off and on:

| Model | Search | Off: MAE / RMSE / WAPE | On: MAE / RMSE / WAPE |
| --- | --- | --- | --- |
| Chronos-2 | Grid | 0.5192 / 0.7975 / 13.65 | 0.5248 / 0.8023 / 13.80 |
| Chronos-2 | Bayesian | 0.5221 / 0.7893 / 13.72 | 0.5262 / 0.8047 / 13.83 |
| Gradient boosting | Grid | 0.5236 / 0.8024 / 13.76 | 0.5212 / 0.8028 / 13.70 |
| Gradient boosting | Bayesian | 0.5378 / 0.8245 / 14.14 | 0.5343 / 0.8203 / 14.04 |

Chronos-2 beat naive by 1.28 percent in grid search, clearing the 1 percent margin, so the publisher registered it as `natgas-price-forecaster` version 8 (`@champion`), the same config as version 7. Bayesian search found Chronos-2 at context 320 without weather, only 0.74 percent better than naive, so on its own it would keep naive.

What the numbers say:

- Over 5 days nothing beats repeating the last price. Over 30 days Chronos-2 pulls ahead, and the gap grows with the horizon: it trails naive on days 1 to 5 and leads on days 16 to 30.
- Chronos-2 does not need weather to win. Without it, context 512 scores MAE 0.5204, 1.05 percent better than naive, with a lower RMSE (0.7774) than the weather config. Weather helps a little, but backtests hand it perfect weather for all 30 days, so treat that gain as an upper bound.
- Yearly terms help gradient boosting a little and hurt Chronos-2. In the grid they lower gradient boosting MAE in 14 of 16 matched configs (0.0046 on average) and the best drops from 0.5236 to 0.5212, now 0.90 percent better than naive, still short of the margin. For Chronos-2 they are worse in all 6 matched configs.
- Holiday dummies did not help Greykite, and Silverkite's trend and seasonality terms drift badly past two weeks.

## How it works

```mermaid
flowchart LR
    yaml[config/tuning.yaml] --> configs[model config classes]
    configs --> optimizer[GridSearchOptimizer or BayesianOptimizer]
    provider[ForecastDataProvider] --> backtester[Backtester]
    optimizer --> backtester
    backtester --> mlflow[MLflow experiment]
    optimizer --> publisher[ModelPublisher]
    publisher --> registry[MLflow model registry]
```

Every model subclasses `AbstractModelWrapper` (`train`, `predict`, `save`, `load`) and takes a config class that subclasses `AbstractConfigWrapper` (defaults, validation, grid, search space, and what the backtester needs). The backtester, optimizers, and publisher work with any of them.

### Code highlights

`ForecastDataProvider` is the only way a model gets data. It hands out a `ForecastWindow` per as-of date, and `Backtester` refuses to run a window that leaks:

```python
@staticmethod
def check_no_leakage(
    as_of: pd.Timestamp, training_data: pd.DataFrame, forecast_dates: pd.DatetimeIndex
) -> None:
    """Stop the run if any training row is after the as-of date."""
    if training_data.index.max() > as_of:
        raise AssertionError(f"Training data goes past the as-of date {as_of.date()}")
    if forecast_dates.min() <= as_of:
        raise AssertionError(f"Forecast dates must start after {as_of.date()}")
```

`ChronosModel` runs Amazon Chronos-2 zero-shot. The context is prices up to the as-of date, and degree days for the forecast dates go in as known future covariates. The median is the point forecast, and the 10th and 90th percentiles make an 80 percent interval:

```python
forecast = pipeline.predict_df(
    self.context_frame(),
    future_df=self.future_covariates(future_features, future_calendar),
    prediction_length=len(future_calendar),
    quantile_levels=[LOWER_QUANTILE, MEDIAN_QUANTILE, UPPER_QUANTILE],
    id_column=ID_COLUMN,
    timestamp_column=TIME_COLUMN,
    target=TARGET_COLUMN,
    context_length=self.config.context_length,
    freq="B",
)
```

`ModelPublisher` only replaces naive when the winner clears the margin:

```python
improvement = 1 - best.metrics()[metric] / baseline.metrics()[metric]
if best.model_name == BASELINE_MODEL:
    return PublishDecision(baseline, "naive had the lowest error", 0.0)
if improvement >= self.min_improvement:
    reason = f"{best.model_name} beat naive by {improvement:.2%}"
    return PublishDecision(best, reason, improvement)
```

### Models

| Model | What it does |
| --- | --- |
| `NaiveModel` | Repeats the last price. The baseline every other model has to beat. |
| `GradientBoostingModel` | scikit-learn gradient boosted trees on recent daily price changes, plus optional degree days and calendar features. Forecasts several days ahead by feeding each prediction back in. |
| `GreykiteModel` | LinkedIn Greykite Silverkite: trend, seasonality, recent prices as lags, and US holiday dummies as regressors. Each holiday gets a dummy for the day itself and for up to 4 days before and after (`labor_day_pre_2`, `labor_day_post_1`). Dummies that are never 1 in the training window are left out. |
| `ChronosModel` | Amazon Chronos-2, a pretrained time series model, used zero-shot on CPU. Degree days and calendar features can be passed as known future covariates. Tuned on context length, weather, and the yearly terms. |

### Data, backtests, search, and publishing

| Class | Job |
| --- | --- |
| `ForecastDataProvider` | Give it an as-of date and it returns a `ForecastWindow`: training data on or before that date, feature values for the dates to predict, and the actual prices used to score the forecast. |
| `AbstractFeatureSource` | Base class for extra data. Each source returns only what was known on the as-of date. `WeatherFeatureSource` adds heating and cooling degree days. `HolidayFeatureSource` adds US holiday dummies from the `holidays` package, which are known in advance and so are valid for the forecast dates. `CalendarFeatureSource` adds Fourier terms for the time of year (2 harmonics) and day of week (1 or 2 harmonics) and an `after_weekend` flag for the first weekday after a weekend; these depend only on the date. Only the yearly terms are tuned. `StorageFeatureSource` adds the EIA storage level once each weekly report is published. |
| `Backtester` | For each as-of date gets the window, checks for leakage, trains a fresh model, predicts, and records a `BacktestResult`. Runs dates in parallel with `n_jobs`. |
| `BacktestReport`, `ForecastMetrics` | Score one config: MAE, RMSE, MAPE, WAPE, and bias, overall, by days ahead, and by horizon bucket (days 1 to 5, 6 to 15, 16 to 30). |
| `AbstractOptimizer` | Backtests candidate configs, logs each one as a nested MLflow run, skips repeats, records failures without stopping, and builds the results table with skill against naive. |
| `GridSearchOptimizer`, `BayesianOptimizer` | Every combination in the grid, or Optuna's TPE sampler picking each next config from the ones already scored. |
| `ModelPublisher` | Publishes the best config only if it lowers error by at least `min_improvement` against naive (1 percent by default). The chosen config is retrained on all data, logged as an MLflow pyfunc model, registered, and given the `champion` alias. |
| `TuningPipeline` | Runs all of the above from `tuning.yaml`. `natgas tune` calls it. |

To add a model, create a folder under `models/` with a config class and a model class, add the pair to `ModelCatalog`, and give it a grid and search space in `tuning.yaml`. Nothing else changes.

### No look-ahead

- Training data always ends on the as-of date, and the test period starts the next trading day. The backtester checks this on every window.
- Storage values only appear once EIA would have published them (6 days after the week ends).
- Weather for future dates would come from the forecast made on the as-of date. Past forecasts are not stored, so backtests use the observed weather as a perfect forecast. This makes weather models look better than they will live, and much more so at 30 days than at 5: real weather forecasts lose most of their skill after about two weeks, while the backtest still hands the models the exact weather for days 16 to 30. Pass `use_observed_as_forecast=False` to `WeatherFeatureSource` to hide future weather instead. Production scoring cannot see future weather either, so it uses the real forecast and then normals (see below).

## Quickstart

Requires Python 3.10 (Greykite does not support newer versions yet). Greykite 1.1 pins pandas below 2.0, numpy 1.26.0, and scikit-learn 1.3.1. Chronos-2's package metadata asks for pandas 2, but the forecast API used here works with pandas 1.5, so it is installed in a second step that leaves pandas alone. Install the CPU build of PyTorch first on a machine without a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pip install "chronos-forecasting>=2.0" --no-deps

export EIA_API_KEY=your_key   # optional, storage is skipped without it
natgas init-db
natgas ingest all             # 2015-01-01 through today by default
natgas tune                   # backtest, log to MLflow, register the best model
```

More commands:

```bash
natgas ingest prices --start 2024-01-01 --end 2024-06-30
natgas tune --optimizer bayesian     # Bayesian search instead of grid
natgas tune --no-publish             # backtest and log only
mlflow ui --backend-store-uri sqlite:///mlflow.db
make lint && make test
```

Loaders upsert on each table's key, so reruns do not create duplicates. Every run is logged to the `ingestion_runs` table. Run commands from the repo root: MLflow runs go to `mlflow.db` with artifacts in `mlruns/`, both git ignored. Chronos-2 weights are downloaded from Hugging Face the first time and cached; only the model id is saved with a published model. Load the champion anywhere with:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/natgas-price-forecaster@champion")
model.predict(future_dates_frame)  # columns: date plus the features the model uses
```

On an 8 core CPU the grid notebook takes about 25 minutes and the Bayesian notebook about 40, most of it Greykite.

## Production scoring

`natgas score` (or `make score`) is the daily job. Run it by hand, from cron with `scripts/run_scoring.sh`, or as a Prefect flow (see Prefect below).

1. `DataRefresher` reloads the last 30 days of prices, weather, and storage with the same loaders as `natgas ingest`, so late corrections replace earlier values.
2. The as-of date is the latest trading day before today. Today's bar is still trading, so it is never used.
3. `ProductionScorer` loads two versions of `natgas-price-forecaster` from the MLflow registry: `@champion` is scored as stage `production` (Chronos-2, context 512, with weather, version 8) and `@challenger` as stage `staging` (the best grid search gradient boosting config, version 9). Each version's config is refit on data through the as-of date, the same way every backtest ran. The scorer refuses a Chronos-2 config without weather or a Greykite config with weather.
4. `PredictionStore` appends 30 rows per stage to `predictions`. A stage that already has rows for the as-of date is skipped, and the table key (`as_of_date`, `stage`, `horizon_day`) blocks duplicates, so reruns are safe.
5. `ActualsRefresher` copies every forecast whose target date has a price into `forecast_actuals`, with the actual and the error.
6. `ForecastMonitor` recreates the `forecast_accuracy` and `forecast_accuracy_rolling` views: MAE, MAPE, WAPE, and bias by model, stage, and horizon bucket (days 1 to 5, 6 to 15, 16 to 30, and all days), over all matured forecasts and over the last 20 as-of dates.

Future weather. Chronos-2 uses degree days for the 30 forecast days, and those days have not happened yet. `ForecastWeatherFeatureSource` fills them with the Open-Meteo forecast that the weather loader stores (14 days, `weather.forecast_days` in `config/settings.yaml`, which covers about 10 trading days) and, after that, the daily normal: the average degree days for that day of the year over the last 10 years, smoothed over 15 days. The backtests used observed weather for all 30 days, so expect live accuracy on days 11 to 30 to sit closer to Chronos-2 without weather.

Forecast dates past the last price skip exchange holidays (`TradingCalendar`), so every forecast gets an actual.

```bash
natgas register-challenger   # once: best gradient boosting config as @challenger
natgas export-backtests      # backtest_forecasts table and outputs/backtest_forecasts.parquet
natgas score                 # refresh, score, backfill actuals, update the views
natgas score --no-refresh    # score from the data already loaded
```

### Prefect

`ScoringFlow` runs the same job as a Prefect 3 flow, `natgas-daily-scoring`, with one task per stage: `refresh_data` (retries twice, 30 seconds apart, for network hiccups at Yahoo, Open-Meteo, or EIA), `score_models`, and `backfill_actuals`. The tasks call `ScoringPipeline`'s stage methods, so `natgas score` and `natgas flow` write the same rows and reruns stay safe. The open-source Prefect server shows run history, each task's state and logs, failures with tracebacks, and the schedule at http://127.0.0.1:4200.

Run from the repo root. Start the server in one terminal and leave it open:

```bash
prefect server start          # UI at http://127.0.0.1:4200, data in ~/.prefect
```

In a second terminal, point the client at it, then run once or serve the schedule:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api   # once per machine
natgas flow --no-refresh      # one tracked run from the data already loaded
natgas flow                   # one tracked run with a data refresh
natgas serve                  # deployment natgas-daily-scoring, 07:00 Tuesday to Saturday America/New_York
prefect deployment run natgas-daily-scoring/natgas-daily-scoring   # trigger the served deployment now
```

`natgas serve` keeps running and starts each scheduled run itself, at most one at a time; stopping it with Ctrl+C pauses the schedule until the next `natgas serve`. Both the server and `natgas serve` must be running at 07:00, so on Windows start them at logon with Task Scheduler. Without `PREFECT_API_URL`, `natgas flow` starts a temporary server for that run only, and nothing is kept for the UI. The same commands work in PowerShell; use `.venv\Scripts\natgas.exe` if the virtual environment is not activated.

Prefect 3.8 breaks on FastAPI 0.151 and later, so `pyproject.toml` caps FastAPI below 0.151.

`export-backtests` takes the best config per model from the latest 30 day grid search in MLflow and writes the forecasts that search logged. Chronos-2 is backtested again on the same as-of dates only to add its interval, and as-of dates where the rerun does not match the logged forecast (for example after the weather archive revised recent days) get no interval.

Tables and views in `data/natgas.db`:

| Name | Kind | Columns |
| --- | --- | --- |
| `backtest_forecasts` | table, rewritten by the export | as_of_date, target_date, horizon_day, model, model_version, forecast, lower, upper, actual |
| `predictions` | table, append only | run_id, run_timestamp, as_of_date, target_date, horizon_day, model, model_version, stage, forecast, lower, upper |
| `forecast_actuals` | table, matured predictions | the `predictions` columns plus horizon_bucket, actual, error (forecast minus actual), filled_at |
| `forecast_accuracy` | view | model, stage, horizon_bucket, forecasts, as_of_dates, first_as_of_date, last_as_of_date, mae, mape, wape, bias |
| `forecast_accuracy_rolling` | view | model, stage, horizon_bucket, as_of_date, as_of_dates, forecasts, mae, mape, wape, bias |

`lower` and `upper` are the 10th and 90th percentiles, filled for Chronos-2 only. `model_version` is the registry version, or `run-` plus the MLflow run id for a backtest config that was never registered.

## Spark

The SQL database is the system of record and the loaders write to it. `SparkDatabase` in `src/natgas/de/spark_db.py` reads and writes the same tables over JDBC, so later jobs can run on Spark without moving the data.

```bash
pip install -e ".[dev,spark]"
natgas spark-show prices
```

```python
from natgas.config import load_settings
from natgas.de.spark_db import SparkDatabase

database = SparkDatabase(load_settings().db_url)
prices = database.read_table("prices")
database.write_table(prices.select("date", "high"), "prices_high", mode="overwrite")
```

- Needs Java 17, with `JAVA_HOME` set or `java` on the path.
- The JDBC driver (sqlite-jdbc, or the Postgres driver when `NATGAS_DB_URL` is Postgres) is downloaded from Maven Central the first time Spark starts.
- JDBC writes append or overwrite, they do not upsert. Write Spark output to its own tables and leave the loader tables to the loaders.
- Timestamps written from Spark to SQLite keep milliseconds only.
- On Windows, Spark warns at startup that `winutils.exe` and `HADOOP_HOME` are missing. Database reads and writes work without them; they are only needed to write files such as Parquet or CSV.

The Spark tests skip when pyspark or Java is not installed.


## Configuration

Settings live in `config/settings.yaml` (ticker, storage series, cities, degree day base). Environment variables:

| Variable | Purpose |
| --- | --- |
| `NATGAS_DB_URL` | SQLAlchemy URL. Defaults to `sqlite:///data/natgas.db`; a Postgres URL works too |
| `EIA_API_KEY` | Needed for storage data |
| `NATGAS_CONFIG` | Path to a different settings file |

## Architecture

```mermaid
flowchart LR
    yahoo[Yahoo Finance] --> prices[de/prices]
    meteo[Open-Meteo] --> weather[de/weather]
    eia[EIA API] --> storage[de/storage]
    prices --> db[(SQL database)]
    weather --> db
    storage --> db
    db <--> spark[Spark over JDBC]
    db --> models[models and backtests]
    models --> mlflow[MLflow registry]
    mlflow --> forecast[daily forecast]
    forecast --> db
    db --> dashboard[dashboard]
```

Ingestion, backtesting, MLflow tracking, the model registry, and the daily scoring job exist today. The dashboard is planned. The scoring job can run on a schedule with cron or Prefect.

## Layout

```
config/settings.yaml    tickers, series, cities
src/natgas/config.py    settings loader
src/natgas/db/          tables, sessions, upsert
src/natgas/de/          data loading
  prices/               PriceLoader (Yahoo Finance)
  weather/              WeatherLoader (Open-Meteo)
  storage/              StorageLoader (EIA)
  utils.py              Loader base class, run logging, HTTP retries, date ranges
  spark_db.py           SparkDatabase, reads and writes the database over JDBC
src/natgas/forecasting/ models, backtests, search, and publishing
  data/                 ForecastDataProvider, ForecastWindow, SqlReader, TradingCalendar,
                        AbstractFeatureSource, WeatherFeatureSource, HolidayFeatureSource,
                        CalendarFeatureSource, StorageFeatureSource,
                        ForecastWeatherFeatureSource
  models/
    base/               AbstractModelWrapper, AbstractConfigWrapper
    naive/              NaiveModel, NaiveModelConfig
    gradient_boosting/  GradientBoostingModel, GradientBoostingModelConfig
    greykite/           GreykiteModel, GreykiteModelConfig
    chronos/            ChronosModel, ChronosModelConfig
    ModelCatalog.py     name to (model class, config class)
  backtesting/          Backtester, BacktestResult, BacktestReport, ForecastMetrics
  optimization/         AbstractOptimizer, GridSearchOptimizer, BayesianOptimizer,
                        HyperparameterGrid, OptimizerCatalog
  tuning/               TuningConfig, TuningPipeline, TuningOutcome, ModelPublisher,
                        PublishDecision, MlflowModelAdapter
  production/           ScoringPipeline, DataRefresher, ProductionScorer, PredictionStore,
                        RegisteredModelLoader, ActualsRefresher, ForecastMonitor, ScoringRun,
                        GridSearchRuns, BacktestExporter, ChallengerPublisher
config/tuning.yaml      model grids and search spaces, backtest window, publish rule, MLflow
experiments/            grid search and Bayesian search notebooks
src/natgas/cli.py       natgas command
tests/                  pytest, no network calls
```

## Roadmap

- [x] Database and data loaders for prices, weather, and storage
- [x] Spark access to the database over JDBC
- [ ] Data checks for stale, missing, and duplicate data
- [x] Shared model interface (train, predict, save, load) with a config class per model
- [x] Naive, gradient boosting, Greykite, and Chronos-2 models on the same interface
- [ ] ARIMA models
- [x] Rolling-origin backtests and a scoreboard (MAE, RMSE, MAPE, WAPE, bias)
- [x] Grid and Bayesian search logged to MLflow, winner registered as `champion` only if it beats naive
- [ ] Model stages: Testing, Challenger, Champion, Legacy Champion
- [x] Daily scoring job (`natgas score`) for the champion and a challenger, with actuals and accuracy views
- [ ] Run the scoring job on a schedule, and a weekly retrain
- [ ] Streamlit dashboard

## Sources

Data:

- Futures prices: [Yahoo Finance `NG=F`](https://finance.yahoo.com/quote/NG=F/) via [yfinance](https://github.com/ranaroussi/yfinance), daily OHLCV
- Weather: [Open-Meteo](https://open-meteo.com/) archive and forecast APIs, daily mean temperature converted to heating and cooling degree days (base 65F) for 8 cities, no key needed
- Storage: [EIA API v2](https://www.eia.gov/opendata/), weekly Lower 48 working gas in storage, series `NW2_EPG0_SWO_R48_BCF`, [free key](https://www.eia.gov/opendata/register.php)

Models and tools:

- [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821) and the [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) model card
- [Greykite](https://github.com/linkedin/greykite) (Silverkite)
- [holidays](https://github.com/vacanza/holidays) Python package for the US holiday calendar
- [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Optuna](https://optuna.org/) and [MLflow](https://mlflow.org/)

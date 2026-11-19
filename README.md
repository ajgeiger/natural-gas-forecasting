# natural-gas-forecasting

Daily forecasts of the natural gas front-month futures price (Yahoo Finance `NG=F`, daily high), using EIA weekly storage and weather in major gas-demand cities as inputs.

The goal is a small system that runs on its own: pull fresh data each day, keep several models in play, use the most accurate approved one for the published forecast, and track accuracy as actual prices come in.

## Roadmap

- [x] Database and data loaders for prices, weather, and storage
- [x] Spark access to the database over JDBC
- [ ] Data checks for stale, missing, and duplicate data
- [x] Shared model interface (train, predict, save, load) with naive and gradient boosting models
- [ ] ARIMA, Greykite, and Chronos-2 models on the same interface
- [x] Rolling-origin backtests and a scoreboard (MAE, RMSE, MAPE, WAPE, bias)
- [x] Hyperparameter search logged to MLflow, best model registered as `champion`
- [ ] Model stages: Testing, Challenger, Champion, Legacy Champion
- [ ] Daily forecast and weekly retrain jobs in GitHub Actions
- [ ] Streamlit dashboard

## Quickstart

Requires Python 3.10 (Greykite does not support newer versions yet).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export EIA_API_KEY=your_key   # optional, storage is skipped without it
natgas init-db
natgas ingest all             # 2015-01-01 through today by default
natgas tune                   # backtest, log to MLflow, register the best model
```

Each source can also be loaded on its own:

```bash
natgas ingest prices --start 2024-01-01 --end 2024-06-30
natgas ingest weather
natgas ingest storage
```

Loaders upsert on each table's key, so reruns do not create duplicates. Every run is logged to the `ingestion_runs` table.

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

Run `make lint` and `make test` before opening a PR.

## Forecasting framework

The forecasting code lives in `src/natgas/forecasting/`. It is built so every model is trained, backtested, and published the same way, which makes models easy to compare and hard to cheat.

```mermaid
flowchart LR
    config[config/tuning.yaml] --> tuner[HyperparameterTuner]
    provider[ForecastDataProvider] --> backtester[Backtester]
    tuner --> backtester
    backtester --> mlflow[MLflow experiment]
    tuner --> publisher[ModelPublisher]
    publisher --> registry[MLflow model registry]
```

### The pieces

| Class | Job |
| --- | --- |
| `ForecastDataProvider` | The only way models get data. Give it an as-of date and it returns a `ForecastWindow`: training data on or before that date, feature values for the dates to predict, and the actual prices used to score the forecast. |
| `AbstractFeatureSource` | Base class for extra data. Each source returns only what was known on the as-of date. `WeatherFeatureSource` adds heating and cooling degree days. `StorageFeatureSource` adds the EIA storage level once each weekly report is published. |
| `AbstractModelWrapper` | Shared interface: `train`, `predict`, `save`, `load`. Hyperparameters go in the constructor and unknown names are rejected. |
| `NaiveModel` | Repeats the last price. The baseline every model has to beat. |
| `GradientBoostingModel` | scikit-learn gradient boosted trees on recent daily price changes, plus degree days when `use_weather` is on. Forecasts several days ahead by feeding each prediction back in. |
| `Backtester` | Replays history. For each as-of date it gets the window, checks for leakage, trains a fresh model, predicts, and records a `BacktestResult`. |
| `BacktestReport`, `ForecastMetrics` | Collect the results for one config and score them: MAE, RMSE, MAPE, WAPE, and bias, overall and by days ahead. |
| `HyperparameterTuner` | Expands the grid in `tuning.yaml`, backtests every config, logs each one as a nested MLflow run, and builds a summary table with skill against the naive baseline. |
| `ModelPublisher` | Retrains the best config on all data, logs it as an MLflow pyfunc model, registers a new version, and points the `champion` alias at it. |
| `TuningPipeline` | Runs all of the above from one config. `natgas tune` calls it. |

### No look-ahead

- Training data always ends on the as-of date, and the test period starts the next trading day. The backtester checks this on every window and stops if it is broken.
- Storage values only appear once EIA would have published them (6 days after the week ends).
- Weather for future dates would come from the forecast made on the as-of date. Past forecasts are not stored, so backtests use the observed weather as a perfect forecast. This makes weather models look slightly better than they will live. Pass `use_observed_as_forecast=False` to `WeatherFeatureSource` to hide future weather instead.

### Run it

```bash
natgas ingest all
natgas tune                          # uses config/tuning.yaml
natgas tune --no-publish             # backtest and log only
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run these from the repo root. MLflow runs are stored in `mlflow.db` with artifacts in `mlruns/`; both are git ignored. Set `mlflow.tracking_uri` in `tuning.yaml` to use a remote server. Load the published model anywhere with:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/natgas-price-forecaster@champion")
model.predict(future_dates_frame)  # columns: date, hdd, cdd
```

### Adding a model

Subclass `AbstractModelWrapper`, implement the four methods, add the class to `ModelCatalog`, and give it a grid in `tuning.yaml`. The backtester, tuner, and publisher need no changes.

### First results

On prices from 2015 through November 2026, with 127 as-of dates every 14 days from January 2022 and a 5 trading day horizon, 17 configs were tried. The best gradient boosting config (5 lags, depth 2, learning rate 0.03, no weather) scored an MAE of $0.2462 per MMBtu and a WAPE of 6.49%. The naive baseline scored $0.2462 and 6.49%. In other words, the first models tie the baseline, which is normal for a liquid futures price over a few days. The framework is in place to test stronger features and models against that bar.

## Configuration

Settings live in `config/settings.yaml` (ticker, storage series, cities, degree day base). Environment variables:

| Variable | Purpose |
| --- | --- |
| `NATGAS_DB_URL` | SQLAlchemy URL. Defaults to `sqlite:///data/natgas.db`; a Postgres URL works too |
| `EIA_API_KEY` | Needed for storage data |
| `NATGAS_CONFIG` | Path to a different settings file |

## Data sources

| Data | Source | Notes |
| --- | --- | --- |
| Futures prices | [Yahoo Finance](https://finance.yahoo.com/quote/NG=F/) via [yfinance](https://github.com/ranaroussi/yfinance) | Daily OHLCV for `NG=F` |
| Weather | [Open-Meteo](https://open-meteo.com/) archive and forecast APIs | Daily mean temperature, converted to heating and cooling degree days (base 65F) for 8 cities. No key needed |
| Storage | [EIA API v2](https://www.eia.gov/opendata/) | Weekly Lower 48 working gas in storage, series `NW2_EPG0_SWO_R48_BCF`. [Free key](https://www.eia.gov/opendata/register.php) |

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

Ingestion, backtesting, MLflow tracking, and the model registry exist today. The daily forecast job and dashboard are planned.

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
src/natgas/forecasting/ models, backtests, and tuning
  data/                 ForecastDataProvider, ForecastWindow, feature sources
  models/               AbstractModelWrapper, NaiveModel, GradientBoostingModel, ModelCatalog
  backtesting/          Backtester, BacktestResult, BacktestReport, ForecastMetrics
  tuning/               TuningConfig, HyperparameterGrid, HyperparameterTuner, ModelPublisher, TuningPipeline
config/tuning.yaml      hyperparameter grids, backtest window, MLflow settings
src/natgas/cli.py       natgas command
tests/                  pytest, no network calls
```

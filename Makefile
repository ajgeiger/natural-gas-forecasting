install:
	pip install -e ".[dev]"
	pip install "chronos-forecasting>=2.0" --no-deps

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

init-db:
	natgas init-db

ingest:
	natgas ingest all

tune:
	natgas tune

# Refresh data, score production and staging, backfill actuals
score:
	natgas score

# Same job as a Prefect flow, tracked in the Prefect UI
flow:
	natgas flow

prefect-server:
	prefect server start

# Scheduled deployment natgas-daily-scoring, needs prefect-server running
serve:
	natgas serve

export-backtests:
	natgas export-backtests

register-challenger:
	natgas register-challenger

spark-show:
	natgas spark-show prices

.PHONY: install lint format test init-db ingest tune score flow prefect-server serve export-backtests register-challenger spark-show

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

spark-show:
	natgas spark-show prices

.PHONY: install lint format test init-db ingest tune spark-show

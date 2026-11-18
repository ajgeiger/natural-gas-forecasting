install:
	pip install -e ".[dev]"

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

.PHONY: install lint format test init-db ingest

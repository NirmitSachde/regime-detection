.PHONY: help setup sync up down logs test lint fmt typecheck check dbt-build clean ingest train backtest streamlit mlflow

PYTHON := uv run python
UV := uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## One-shot project bootstrap (uv sync + pre-commit install)
	$(UV) sync --all-extras
	$(UV) run pre-commit install

sync:  ## Re-resolve and install dependencies
	$(UV) sync --all-extras

up:  ## Bring up docker-compose stack (Streamlit + MLflow + Prefect)
	docker compose up -d --build
	@echo "Streamlit:  http://localhost:8501"
	@echo "MLflow:     http://localhost:5000"
	@echo "Prefect:    http://localhost:4200"

down:  ## Tear down docker-compose stack
	docker compose down

logs:  ## Tail container logs
	docker compose logs -f

test:  ## Run pytest with coverage
	$(UV) run pytest

test-fast:  ## Run pytest without coverage gate (quick feedback)
	$(UV) run pytest --no-cov -x

lint:  ## Ruff check
	$(UV) run ruff check .

fmt:  ## Ruff format
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## Mypy strict on src/
	$(UV) run mypy src

check: lint typecheck test  ## Run all quality gates

dbt-build:  ## Build dbt models
	cd dbt && $(UV) run dbt build --profiles-dir .

dbt-docs:  ## Generate and serve dbt docs
	cd dbt && $(UV) run dbt docs generate --profiles-dir . && $(UV) run dbt docs serve --profiles-dir .

ingest:  ## Run ingestion flows (yfinance + FRED)
	$(UV) run regime-ingest all

train:  ## Train both regime models
	$(UV) run regime-train hmm
	$(UV) run regime-train lgbm

backtest:  ## Run all backtests
	$(UV) run regime-backtest all

streamlit:  ## Run Streamlit dashboard locally
	$(UV) run streamlit run src/regime/app/streamlit_app.py

mlflow:  ## Run MLflow UI locally
	$(UV) run mlflow ui --backend-store-uri ./data/mlruns --host 0.0.0.0 --port 5000

api:  ## Run FastAPI in dev mode (auto-reload)
	$(UV) run uvicorn regime.api.main:app --reload --host 0.0.0.0 --port 8000

docs:  ## Build code reference with pdoc into ./docs-site/
	./scripts/build_docs.sh docs-site

docs-serve: docs  ## Build and serve docs locally
	python3 -m http.server --directory docs-site 8088

clean:  ## Remove caches, build artifacts (keeps data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +

clean-data:  ## DESTRUCTIVE: remove all generated data (raw, warehouse, mlruns)
	rm -rf data/raw data/warehouse.duckdb data/warehouse.duckdb.wal data/backtests data/mlruns
	mkdir -p data/raw data/backtests data/mlruns

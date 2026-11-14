.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

sync: ## Install all dependencies with uv (client + server + xarray extras)
	uv sync --all-extras

sync-gdal: ## Pin gdal override to the installed system libgdal version, then sync
	@GDAL_VERSION=$$(gdal-config --version 2>/dev/null) && \
		[ -n "$$GDAL_VERSION" ] || (echo "gdal-config not found — install libgdal-dev first" && exit 1) && \
		echo "System GDAL: $$GDAL_VERSION" && \
		sed -i.bak "s/\"gdal==[^\"]*\"/\"gdal==$$GDAL_VERSION\"/" pyproject.toml && \
		rm -f pyproject.toml.bak && \
		uv sync --all-extras

run: ## Start the app with uvicorn (honors PORT, default 9000)
	uv run uvicorn open_climate_service.main:app --port $${PORT:-9000} --reload --reload-include "*.html" --reload-include "*.css" --reload-include "*.js" --reload-include "*.yaml" --reload-include "*.yml"

lint: ## Check linting, formatting, and types (no autofix)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy open_climate_service/
	uv run pyright

fix: ## Autofix ruff lint and format issues
	uv run ruff check --fix .
	uv run ruff format .

PYTEST_WORKERS ?= 2

test: ## Run tests with pytest in parallel (PYTEST_WORKERS, default 2)
	uv run pytest -n $(PYTEST_WORKERS) tests/

test-serial: ## Run tests with pytest in one process, to diagnose a failure
	uv run pytest tests/

test-integration: ## Fetch from every built-in data source, one at a time; needs every source's key
	uv run pytest -m integration tests/

# Compose bind-mounts this into the container; a missing file would make Docker
# create a directory at the mount point, so fail early with guidance instead.
climate-service.yaml:
	@echo "ERROR: climate-service.yaml is missing. Create it from the example:" >&2
	@echo "  cp climate-service.yaml.example climate-service.yaml" >&2
	@exit 1

start: climate-service.yaml ## Start the Docker stack (builds images first)
	docker compose up --build --remove-orphans

restart: climate-service.yaml ## Tear down, rebuild, and start the Docker stack from scratch
	docker compose down -v && docker compose build --no-cache && docker compose up --remove-orphans

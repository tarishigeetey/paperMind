.PHONY: help start stop restart status logs health setup format lint test test-cov clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

start: ## Build and start all services
	docker compose -f compose.yml up --build -d

stop: ## Stop all services
	docker compose -f compose.yml down

restart: ## Restart all services
	docker compose -f compose.yml restart

status: ## Show service status
	docker compose -f compose.yml ps

logs: ## Stream all logs
	docker compose -f compose.yml logs -f

health: ## Check all services
	@echo "Checking services..."
	@curl -s http://localhost:8000/api/v1/ping | python3 -m json.tool || echo "API down"
	@curl -s http://localhost:9200/_cluster/health | python3 -m json.tool || echo "OpenSearch down"
	@curl -s http://localhost:8080/health || echo "Airflow down"
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool || echo "Ollama down"

setup: ## Install dependencies
	uv sync

format: ## Format code with ruff
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=html

clean: ## Remove containers and volumes (destroys data)
	docker compose -f compose.yml down -v
	docker system prune -f

# =============================================================================
# Oasis Crypto Trade - Makefile
# =============================================================================
# Comprehensive automation for development, testing, and deployment
# 
# Author: Oasis Trading Systems
# License: Proprietary
# =============================================================================

# Default shell
SHELL := /bin/bash

# Project configuration
PROJECT_NAME := oasis-crypto-trade
VERSION := 1.0.0
PYTHON_VERSION := 3.11

# Directories
SRC_DIR := .
APPS_DIR := apps
LIBS_DIR := libs
TOOLS_DIR := tools
DOCS_DIR := docs
INFRA_DIR := infra
DOCKER_DIR := docker
TESTS_DIR := tests

# Python and Poetry
PYTHON := python3.11
POETRY := poetry
POETRY_RUN := $(POETRY) run
PIP := pip3

# Docker
DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_REGISTRY := oasis-crypto-trade

# Testing
PYTEST := $(POETRY_RUN) pytest
PYTEST_COV := $(PYTEST) --cov=apps --cov=libs --cov-report=html --cov-report=term
PYTEST_ARGS := -v --tb=short

# Code quality
BLACK := $(POETRY_RUN) black
ISORT := $(POETRY_RUN) isort
FLAKE8 := $(POETRY_RUN) flake8
MYPY := $(POETRY_RUN) mypy
BANDIT := $(POETRY_RUN) bandit
SAFETY := $(POETRY_RUN) safety

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# =============================================================================
# HELP AND INFO
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║                    $(WHITE)OASIS CRYPTO TRADE                        $(CYAN)║$(RESET)"
	@echo "$(CYAN)║                Enterprise Trading System                       ║$(RESET)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(BLUE)%-20s$(RESET) %s\n", "Command", "Description"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2 } \
		/^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

.PHONY: info
info: ## Show project information
	@echo "$(CYAN)Project Information:$(RESET)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Python: $(PYTHON_VERSION)"
	@echo "  Poetry: $(shell $(POETRY) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Docker: $(shell $(DOCKER) --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "$(CYAN)Directory Structure:$(RESET)"
	@echo "  Apps: $(APPS_DIR)"
	@echo "  Libraries: $(LIBS_DIR)"
	@echo "  Tools: $(TOOLS_DIR)"
	@echo "  Documentation: $(DOCS_DIR)"
	@echo "  Infrastructure: $(INFRA_DIR)"

##@ 🚀 Development

.PHONY: install
install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(POETRY) install
	$(POETRY_RUN) pre-commit install
	@echo "$(GREEN)✅ Dependencies installed$(RESET)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(POETRY) install --with dev
	$(POETRY_RUN) pre-commit install
	@echo "$(GREEN)✅ Development dependencies installed$(RESET)"

.PHONY: update
update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	$(POETRY) update
	@echo "$(GREEN)✅ Dependencies updated$(RESET)"

.PHONY: env
env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from template...$(RESET)"; \
		cp .env.example .env; \
		echo "$(GREEN)✅ .env file created. Please update it with your settings.$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  .env file already exists$(RESET)"; \
	fi

.PHONY: setup
setup: install env ## Complete development setup
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	$(MAKE) check-system
	$(MAKE) generate-secrets
	@echo "$(GREEN)✅ Development environment setup complete$(RESET)"

.PHONY: generate-secrets
generate-secrets: ## Generate application secrets
	@echo "$(GREEN)Generating application secrets...$(RESET)"
	$(POETRY_RUN) python tools/scripts/generate_secrets.py
	@echo "$(GREEN)✅ Secrets generated$(RESET)"

.PHONY: check-system
check-system: ## Check system requirements
	@echo "$(GREEN)Checking system requirements...$(RESET)"
	@$(PYTHON) --version || (echo "$(RED)❌ Python $(PYTHON_VERSION) not found$(RESET)" && exit 1)
	@$(POETRY) --version >/dev/null 2>&1 || (echo "$(RED)❌ Poetry not found. Install with: curl -sSL https://install.python-poetry.org | python3 -$(RESET)" && exit 1)
	@$(DOCKER) --version >/dev/null 2>&1 || echo "$(YELLOW)⚠️  Docker not found (optional for local development)$(RESET)"
	@$(DOCKER_COMPOSE) --version >/dev/null 2>&1 || echo "$(YELLOW)⚠️  Docker Compose not found (optional for local development)$(RESET)"
	@echo "$(GREEN)✅ System requirements check complete$(RESET)"

##@ 🧪 Testing

.PHONY: test
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	$(PYTEST_COV) $(PYTEST_ARGS)
	@echo "$(GREEN)✅ All tests completed$(RESET)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	$(PYTEST) -m "unit" $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Unit tests completed$(RESET)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	$(PYTEST) -m "integration" $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running end-to-end tests...$(RESET)"
	$(PYTEST) -m "e2e" $(PYTEST_ARGS)
	@echo "$(GREEN)✅ End-to-end tests completed$(RESET)"

.PHONY: test-trading
test-trading: ## Run trading-specific tests
	@echo "$(GREEN)Running trading tests...$(RESET)"
	$(PYTEST) -m "trading" $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Trading tests completed$(RESET)"

.PHONY: test-fast
test-fast: ## Run fast tests (exclude slow tests)
	@echo "$(GREEN)Running fast tests...$(RESET)"
	$(PYTEST) -m "not slow" $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Fast tests completed$(RESET)"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(RESET)"
	$(PYTEST) --looponfail $(PYTEST_ARGS)

.PHONY: coverage
coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(RESET)"
	$(PYTEST_COV) --cov-report=html
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(RESET)"

.PHONY: coverage-open
coverage-open: coverage ## Open coverage report in browser
	@python -m webbrowser htmlcov/index.html

##@ 🔧 Code Quality

.PHONY: format
format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	$(ISORT) .
	$(BLACK) .
	@echo "$(GREEN)✅ Code formatted$(RESET)"

.PHONY: lint
lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(RESET)"
	$(MAKE) lint-flake8
	$(MAKE) lint-mypy
	$(MAKE) lint-bandit
	@echo "$(GREEN)✅ All linters completed$(RESET)"

.PHONY: lint-flake8
lint-flake8: ## Run flake8 linter
	@echo "$(GREEN)Running flake8...$(RESET)"
	$(FLAKE8) apps/ libs/ tools/

.PHONY: lint-mypy
lint-mypy: ## Run mypy type checker
	@echo "$(GREEN)Running mypy...$(RESET)"
	$(MYPY) apps/ libs/ tools/

.PHONY: lint-bandit
lint-bandit: ## Run bandit security linter
	@echo "$(GREEN)Running bandit...$(RESET)"
	$(BANDIT) -r apps/ libs/ tools/ -f json

.PHONY: security
security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(RESET)"
	$(SAFETY) check
	$(BANDIT) -r apps/ libs/ tools/
	@echo "$(GREEN)✅ Security checks completed$(RESET)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(RESET)"
	$(POETRY_RUN) pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit hooks completed$(RESET)"

.PHONY: check
check: lint test ## Run all code quality checks
	@echo "$(GREEN)✅ All code quality checks completed$(RESET)"

##@ 🐳 Docker

.PHONY: docker-build
docker-build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✅ Docker images built$(RESET)"

.PHONY: docker-up
docker-up: ## Start Docker infrastructure
	@echo "$(GREEN)Starting Docker infrastructure...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Docker infrastructure started$(RESET)"

.PHONY: docker-down
docker-down: ## Stop Docker infrastructure
	@echo "$(GREEN)Stopping Docker infrastructure...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ Docker infrastructure stopped$(RESET)"

.PHONY: docker-logs
docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

.PHONY: docker-ps
docker-ps: ## Show Docker container status
	$(DOCKER_COMPOSE) ps

.PHONY: docker-clean
docker-clean: ## Clean Docker resources
	@echo "$(GREEN)Cleaning Docker resources...$(RESET)"
	$(DOCKER_COMPOSE) down -v --rmi all --remove-orphans
	$(DOCKER) system prune -f
	@echo "$(GREEN)✅ Docker resources cleaned$(RESET)"

##@ 🚀 Application

.PHONY: run-trading-engine
run-trading-engine: ## Run trading engine service
	@echo "$(GREEN)Starting trading engine...$(RESET)"
	$(POETRY_RUN) python -m apps.trading_engine.main

.PHONY: run-market-data
run-market-data: ## Run market data service
	@echo "$(GREEN)Starting market data service...$(RESET)"
	$(POETRY_RUN) python -m apps.market_data_service.main

.PHONY: run-risk-management
run-risk-management: ## Run risk management service
	@echo "$(GREEN)Starting risk management service...$(RESET)"
	$(POETRY_RUN) python -m apps.risk_management.main

.PHONY: run-analytics
run-analytics: ## Run analytics service
	@echo "$(GREEN)Starting analytics service...$(RESET)"
	$(POETRY_RUN) python -m apps.analytics_service.main

.PHONY: run-dashboard
run-dashboard: ## Run web dashboard
	@echo "$(GREEN)Starting web dashboard...$(RESET)"
	cd apps/web-dashboard && npm start

##@ 🗃️ Database

.PHONY: db-upgrade
db-upgrade: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(RESET)"
	$(POETRY_RUN) alembic upgrade head
	@echo "$(GREEN)✅ Database migrations completed$(RESET)"

.PHONY: db-downgrade
db-downgrade: ## Rollback database migration
	@echo "$(YELLOW)Rolling back database migration...$(RESET)"
	$(POETRY_RUN) alembic downgrade -1

.PHONY: db-revision
db-revision: ## Create new database migration
	@read -p "Enter migration message: " message; \
	$(POETRY_RUN) alembic revision --autogenerate -m "$$message"

.PHONY: db-reset
db-reset: ## Reset database (⚠️  DESTRUCTIVE)
	@echo "$(RED)⚠️  This will destroy all data!$(RESET)"
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(POETRY_RUN) alembic downgrade base; \
		$(POETRY_RUN) alembic upgrade head; \
		echo "$(GREEN)✅ Database reset completed$(RESET)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(RESET)"; \
	fi

.PHONY: db-seed
db-seed: ## Seed database with test data
	@echo "$(GREEN)Seeding database...$(RESET)"
	$(POETRY_RUN) python tools/scripts/seed_database.py
	@echo "$(GREEN)✅ Database seeded$(RESET)"

.PHONY: db-backup
db-backup: ## Backup database
	@echo "$(GREEN)Creating database backup...$(RESET)"
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	$(DOCKER_COMPOSE) exec -T oasis-postgres pg_dump -U oasis_admin oasis_trading_db > "backup_$$timestamp.sql"; \
	echo "$(GREEN)✅ Database backup created: backup_$$timestamp.sql$(RESET)"

##@ 📊 Monitoring

.PHONY: logs
logs: ## View application logs
	tail -f logs/*.log

.PHONY: metrics
metrics: ## Show application metrics
	@echo "$(GREEN)Application Metrics:$(RESET)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

.PHONY: health
health: ## Check service health
	@echo "$(GREEN)Checking service health...$(RESET)"
	@curl -s http://localhost:8000/health | jq '.' || echo "Trading Engine: Offline"
	@curl -s http://localhost:8001/health | jq '.' || echo "Market Data: Offline"
	@curl -s http://localhost:8002/health | jq '.' || echo "Risk Management: Offline"
	@curl -s http://localhost:8003/health | jq '.' || echo "Analytics: Offline"

##@ 📚 Documentation

.PHONY: docs
docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(RESET)"
	$(POETRY_RUN) mkdocs build
	@echo "$(GREEN)✅ Documentation generated$(RESET)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(RESET)"
	$(POETRY_RUN) mkdocs serve

.PHONY: docs-api
docs-api: ## Generate API documentation
	@echo "$(GREEN)Generating API documentation...$(RESET)"
	$(POETRY_RUN) python tools/scripts/generate_api_docs.py
	@echo "$(GREEN)✅ API documentation generated$(RESET)"

##@ 🧹 Cleanup

.PHONY: clean
clean: ## Clean temporary files
	@echo "$(GREEN)Cleaning temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)✅ Temporary files cleaned$(RESET)"

.PHONY: clean-all
clean-all: clean docker-clean ## Clean everything including Docker
	@echo "$(GREEN)✅ Everything cleaned$(RESET)"

##@ 🔧 Utilities

.PHONY: shell
shell: ## Open Python shell with project context
	$(POETRY_RUN) python

.PHONY: jupyter
jupyter: ## Start Jupyter notebook
	$(POETRY_RUN) jupyter notebook

.PHONY: profile
profile: ## Run performance profiler
	@echo "$(GREEN)Running performance profiler...$(RESET)"
	$(POETRY_RUN) py-spy top --pid $$(pgrep -f "trading_engine")

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	$(PYTEST) tests/benchmarks/ -v --benchmark-only

##@ 🚢 Deployment

.PHONY: build
build: ## Build application for production
	@echo "$(GREEN)Building application...$(RESET)"
	$(POETRY) build
	@echo "$(GREEN)✅ Application built$(RESET)"

.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(RESET)"
	# Add staging deployment commands here
	@echo "$(GREEN)✅ Deployed to staging$(RESET)"

.PHONY: deploy-prod
deploy-prod: ## Deploy to production (⚠️  CAREFUL!)
	@echo "$(RED)⚠️  Deploying to PRODUCTION!$(RESET)"
	@read -p "Type 'DEPLOY' to confirm: " confirm; \
	if [ "$$confirm" = "DEPLOY" ]; then \
		# Add production deployment commands here
		echo "$(GREEN)✅ Deployed to production$(RESET)"; \
	else \
		echo "$(YELLOW)Production deployment cancelled$(RESET)"; \
	fi

##@ 🔄 CI/CD

.PHONY: ci-check
ci-check: install check test ## Run CI checks locally
	@echo "$(GREEN)✅ CI checks completed successfully$(RESET)"

.PHONY: ci-build
ci-build: ## Build for CI
	@echo "$(GREEN)Building for CI...$(RESET)"
	$(MAKE) docker-build
	@echo "$(GREEN)✅ CI build completed$(RESET)"

##@ 📈 Analysis

.PHONY: complexity
complexity: ## Analyze code complexity
	@echo "$(GREEN)Analyzing code complexity...$(RESET)"
	$(POETRY_RUN) radon cc apps/ libs/ -a
	$(POETRY_RUN) radon mi apps/ libs/

.PHONY: dependencies
dependencies: ## Show dependency tree
	$(POETRY) show --tree

.PHONY: outdated
outdated: ## Show outdated dependencies
	$(POETRY) show --outdated

.PHONY: vulnerabilities
vulnerabilities: ## Check for security vulnerabilities
	$(SAFETY) check
	$(POETRY_RUN) pip-audit

# =============================================================================
# HELPER TARGETS
# =============================================================================

.PHONY: _check-poetry
_check-poetry:
	@$(POETRY) --version >/dev/null 2>&1 || (echo "$(RED)❌ Poetry not installed$(RESET)" && exit 1)

.PHONY: _check-docker
_check-docker:
	@$(DOCKER) --version >/dev/null 2>&1 || (echo "$(RED)❌ Docker not installed$(RESET)" && exit 1)

# Default target
.DEFAULT_GOAL := help
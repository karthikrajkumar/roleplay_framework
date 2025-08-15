# Makefile for AI Roleplay Platform Backend

# Variables
DOCKER_COMPOSE = docker-compose
PYTHON = python3
PIP = pip3
PROJECT_NAME = roleplay-platform

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

.PHONY: help install test clean build up down logs restart health format lint security

help: ## Show this help message
	@echo "AI Roleplay Platform Backend - Development Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
install: ## Install dependencies for all services
	@echo "$(YELLOW)Installing dependencies for all services...$(NC)"
	@for service in services/*/; do \
		if [ -f "$$service/requirements.txt" ]; then \
			echo "Installing dependencies for $$service"; \
			cd "$$service" && $(PIP) install -r requirements.txt && cd ../..; \
		fi \
	done
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	@for service in services/*/; do \
		if [ -f "$$service/requirements-dev.txt" ]; then \
			echo "Installing dev dependencies for $$service"; \
			cd "$$service" && $(PIP) install -r requirements-dev.txt && cd ../..; \
		fi \
	done
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

# Environment Setup
setup-env: ## Copy environment file and setup initial configuration
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)Created .env file from .env.example$(NC)"; \
		echo "$(YELLOW)Please update the .env file with your configuration$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

# Database Operations
db-init: ## Initialize database with migrations
	@echo "$(YELLOW)Initializing database...$(NC)"
	@$(DOCKER_COMPOSE) exec user-management alembic upgrade head
	@echo "$(GREEN)Database initialized successfully!$(NC)"

db-migrate: ## Create new database migration
	@echo "$(YELLOW)Creating new migration...$(NC)"
	@$(DOCKER_COMPOSE) exec user-management alembic revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)Migration created successfully!$(NC)"

db-upgrade: ## Apply database migrations
	@echo "$(YELLOW)Applying database migrations...$(NC)"
	@$(DOCKER_COMPOSE) exec user-management alembic upgrade head
	@echo "$(GREEN)Migrations applied successfully!$(NC)"

db-downgrade: ## Downgrade database migration
	@echo "$(YELLOW)Downgrading database...$(NC)"
	@$(DOCKER_COMPOSE) exec user-management alembic downgrade -1
	@echo "$(GREEN)Database downgraded successfully!$(NC)"

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "$(RED)WARNING: This will delete all data! Are you sure? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		$(DOCKER_COMPOSE) up -d postgres redis; \
		sleep 10; \
		$(DOCKER_COMPOSE) exec user-management alembic upgrade head; \
		echo "$(GREEN)Database reset successfully!$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

# Docker Operations
build: ## Build all Docker images
	@echo "$(YELLOW)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker images built successfully!$(NC)"

build-no-cache: ## Build all Docker images without cache
	@echo "$(YELLOW)Building Docker images without cache...$(NC)"
	@$(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)Docker images built successfully!$(NC)"

up: ## Start all services
	@echo "$(YELLOW)Starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)All services started!$(NC)"
	@echo "$(YELLOW)Waiting for services to be ready...$(NC)"
	@sleep 15
	@make health

up-build: ## Build and start all services
	@echo "$(YELLOW)Building and starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d --build
	@echo "$(GREEN)All services built and started!$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped!$(NC)"

down-volumes: ## Stop all services and remove volumes
	@echo "$(RED)WARNING: This will delete all persistent data! Are you sure? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		echo "$(GREEN)All services and volumes removed!$(NC)"; \
	else \
		echo "$(YELLOW)Operation cancelled$(NC)"; \
	fi

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	@$(DOCKER_COMPOSE) restart
	@echo "$(GREEN)All services restarted!$(NC)"

restart-service: ## Restart a specific service (usage: make restart-service SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable (e.g., make restart-service SERVICE=api-gateway)$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restarting $(SERVICE)...$(NC)"
	@$(DOCKER_COMPOSE) restart $(SERVICE)
	@echo "$(GREEN)$(SERVICE) restarted!$(NC)"

# Logging and Monitoring
logs: ## Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-service: ## Show logs from a specific service (usage: make logs-service SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable (e.g., make logs-service SERVICE=api-gateway)$(NC)"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) logs -f $(SERVICE)

health: ## Check health of all services
	@echo "$(YELLOW)Checking service health...$(NC)"
	@services=("api-gateway:8000" "user-management:8001" "roleplay-service:8002" "ai-orchestration:8003" "notification-service:8004" "analytics-service:8005"); \
	for service in "$${services[@]}"; do \
		name=$${service%%:*}; \
		port=$${service##*:}; \
		echo -n "Checking $$name... "; \
		if curl -sf http://localhost:$$port/health >/dev/null 2>&1; then \
			echo "$(GREEN)✓ Healthy$(NC)"; \
		else \
			echo "$(RED)✗ Unhealthy$(NC)"; \
		fi; \
	done

# Testing
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@for service in services/*/; do \
		if [ -f "$$service/pytest.ini" ] || [ -f "$$service/pyproject.toml" ]; then \
			echo "Running tests for $$service"; \
			cd "$$service" && $(PYTHON) -m pytest -v && cd ../..; \
		fi \
	done
	@echo "$(GREEN)All tests completed!$(NC)"

test-service: ## Run tests for a specific service (usage: make test-service SERVICE=user-management)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running tests for $(SERVICE)...$(NC)"
	@cd services/$(SERVICE) && $(PYTHON) -m pytest -v

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@for service in services/*/; do \
		if [ -f "$$service/pytest.ini" ] || [ -f "$$service/pyproject.toml" ]; then \
			echo "Running coverage for $$service"; \
			cd "$$service" && $(PYTHON) -m pytest --cov=src --cov-report=html && cd ../..; \
		fi \
	done
	@echo "$(GREEN)Coverage reports generated!$(NC)"

# Code Quality
format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	@for service in services/*/; do \
		echo "Formatting $$service"; \
		cd "$$service" && black src/ && isort src/ && cd ../..; \
	done
	@echo "$(GREEN)Code formatted successfully!$(NC)"

lint: ## Run linting with flake8 and mypy
	@echo "$(YELLOW)Running linting...$(NC)"
	@for service in services/*/; do \
		echo "Linting $$service"; \
		cd "$$service" && flake8 src/ && mypy src/ && cd ../..; \
	done
	@echo "$(GREEN)Linting completed!$(NC)"

security: ## Run security checks with bandit
	@echo "$(YELLOW)Running security checks...$(NC)"
	@for service in services/*/; do \
		echo "Security check for $$service"; \
		cd "$$service" && bandit -r src/ && cd ../..; \
	done
	@echo "$(GREEN)Security checks completed!$(NC)"

# Utility Commands
clean: ## Clean up Docker images and containers
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	@docker system prune -f
	@docker volume prune -f
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: ## Clean up everything including images
	@echo "$(RED)WARNING: This will remove all Docker images and containers! Are you sure? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		docker system prune -af; \
		docker volume prune -f; \
		echo "$(GREEN)Complete cleanup finished!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled$(NC)"; \
	fi

ps: ## Show running containers
	@$(DOCKER_COMPOSE) ps

exec: ## Execute command in a service container (usage: make exec SERVICE=api-gateway CMD="bash")
	@if [ -z "$(SERVICE)" ] || [ -z "$(CMD)" ]; then \
		echo "$(RED)Please specify SERVICE and CMD variables$(NC)"; \
		echo "Example: make exec SERVICE=api-gateway CMD=\"bash\""; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) exec $(SERVICE) $(CMD)

shell: ## Open bash shell in a service container (usage: make shell SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) exec $(SERVICE) bash

# Development Workflow
dev: setup-env up-build db-init ## Complete development setup (setup env, build, start, init db)
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)API Gateway: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Documentation: http://localhost:8000/docs$(NC)"

# Production Commands
prod-build: ## Build production images
	@echo "$(YELLOW)Building production images...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml build
	@echo "$(GREEN)Production images built!$(NC)"

prod-up: ## Start production services
	@echo "$(YELLOW)Starting production services...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)Production services started!$(NC)"
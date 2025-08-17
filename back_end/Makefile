# Makefile for AI Roleplay Platform Backend - Kubernetes Native

# Variables
KUBECTL = kubectl
PYTHON = python3
PIP = pip3
PROJECT_NAME = roleplay-platform
NAMESPACE = roleplay-platform
KUBE_CONTEXT = $(shell kubectl config current-context)

# Kubernetes Development Environment
K8S_CLUSTER_TYPE ?= kind  # kind, minikube, or k3s
K8S_LOCAL_REGISTRY ?= localhost:5001
K8S_NODE_PORT_BASE ?= 30000

# Docker Registry for Development
REGISTRY ?= $(K8S_LOCAL_REGISTRY)
TAG ?= latest

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Service list from architecture
SERVICES = api-gateway user-management ai-orchestration notification analytics real-time-communication

.PHONY: help install test clean build up down logs restart health format lint security \
        k8s-setup k8s-build k8s-deploy k8s-dev k8s-health k8s-logs k8s-cleanup k8s-namespace k8s-secrets \
        k8s-build-service k8s-deploy-dev k8s-logs-service k8s-port-forward k8s-shell k8s-restart \
        k8s-restart-service k8s-status k8s-db-init k8s-db-migrate k8s-db-upgrade k8s-db-reset \
        k8s-stop k8s-clean k8s-clean-all

help: ## Show this help message
	@echo "$(CYAN)AI Roleplay Platform Backend - Kubernetes Native Development$(NC)"
	@echo "=============================================================="
	@echo "$(BLUE)Current Kubernetes Context: $(KUBE_CONTEXT)$(NC)"
	@echo "$(BLUE)Target Namespace: $(NAMESPACE)$(NC)"
	@echo "$(BLUE)Registry: $(REGISTRY)$(NC)"
	@echo ""
	@echo "$(YELLOW)Kubernetes Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^k8s-[a-zA-Z_-]+:.*?## / {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Traditional Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && !/^k8s-/ {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. $(CYAN)make k8s-dev$(NC)        - Complete development setup (first time)"
	@echo "  2. $(CYAN)make k8s-reset-dev$(NC)  - Complete reset and rebuild (destroys everything)"
	@echo "  3. $(CYAN)make k8s-health$(NC)     - Check all services"
	@echo "  4. $(CYAN)make k8s-logs$(NC)       - View all service logs"

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

# =======================================
# Kubernetes Development Commands
# =======================================

k8s-setup: ## Setup local Kubernetes cluster (kind/minikube/k3s)
	@echo "$(YELLOW)Setting up local Kubernetes cluster ($(K8S_CLUSTER_TYPE))...$(NC)"
	@./scripts/k8s-setup.sh $(K8S_CLUSTER_TYPE)
	@echo "$(GREEN)Kubernetes cluster setup completed!$(NC)"

k8s-dev: k8s-setup k8s-namespace k8s-secrets k8s-build k8s-deploy-dev k8s-db-init ## Complete development setup
	@echo "$(GREEN)Kubernetes development environment ready!$(NC)"
	@echo "$(YELLOW)Services are accessible via NodePort:$(NC)"
	@echo "$(CYAN)  API Gateway: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 0)))/health/$(NC)"
	@echo "$(CYAN)  Documentation: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 0)))/docs$(NC)"
	@echo "$(CYAN)  Grafana: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 10)))$(NC)"

k8s-reset-dev: ## Complete reset and rebuild - destroys everything and recreates from scratch
	@echo "$(RED)WARNING: This will completely destroy and recreate the entire development environment!$(NC)"
	@echo "$(RED)This includes: cluster, registry, networks, images, and all data! Continue? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		echo "$(YELLOW)Step 1/5: Cleaning up existing cluster and resources...$(NC)"; \
		./scripts/k8s-cleanup.sh $(K8S_CLUSTER_TYPE) || true; \
		docker system prune -af --volumes || true; \
		echo "$(YELLOW)Step 2/5: Setting up fresh Kubernetes cluster...$(NC)"; \
		./scripts/k8s-setup.sh $(K8S_CLUSTER_TYPE); \
		echo "$(YELLOW)Step 3/5: Creating namespace and secrets...$(NC)"; \
		$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -; \
		./scripts/k8s-create-secrets.sh $(NAMESPACE); \
		echo "$(YELLOW)Step 4/5: Building and pushing all images...$(NC)"; \
		./scripts/build-images.sh $(REGISTRY) latest; \
		echo "$(YELLOW)Step 5/5: Deploying services and initializing database...$(NC)"; \
		./scripts/k8s-deploy-dev.sh $(NAMESPACE) $(REGISTRY) latest $(K8S_NODE_PORT_BASE); \
		$(KUBECTL) exec -n $(NAMESPACE) deployment/user-management -- alembic upgrade head 2>/dev/null || echo "Database init skipped (service not ready)"; \
		echo "$(GREEN)Complete reset and rebuild finished!$(NC)"; \
		echo "$(YELLOW)Services are accessible via NodePort:$(NC)"; \
		echo "$(CYAN)  API Gateway: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 0)))/health/$(NC)"; \
		echo "$(CYAN)  Documentation: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 0)))/docs$(NC)"; \
		echo "$(CYAN)  Grafana: http://localhost:$(shell echo $$(($(K8S_NODE_PORT_BASE) + 10)))$(NC)"; \
	else \
		echo "$(YELLOW)Reset cancelled$(NC)"; \
	fi

k8s-namespace: ## Create Kubernetes namespace
	@echo "$(YELLOW)Creating namespace $(NAMESPACE)...$(NC)"
	@$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@echo "$(GREEN)Namespace created/updated successfully!$(NC)"

k8s-secrets: ## Create Kubernetes secrets
	@echo "$(YELLOW)Creating Kubernetes secrets...$(NC)"
	@./scripts/k8s-create-secrets.sh $(NAMESPACE)
	@echo "$(GREEN)Secrets created successfully!$(NC)"

k8s-build: ## Build and push Docker images for all services
	@echo "$(YELLOW)Building and pushing Docker images...$(NC)"
	@./scripts/build-images.sh $(REGISTRY) $(TAG)
	@echo "$(GREEN)All images built and pushed successfully!$(NC)"

k8s-build-service: ## Build specific service (usage: make k8s-build-service SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Building $(SERVICE)...$(NC)"
	@docker build -t $(REGISTRY)/$(PROJECT_NAME)-$(SERVICE):$(TAG) ./services/$(SERVICE)/
	@docker push $(REGISTRY)/$(PROJECT_NAME)-$(SERVICE):$(TAG)
	@echo "$(GREEN)$(SERVICE) built and pushed successfully!$(NC)"

k8s-deploy: ## Deploy production Kubernetes manifests
	@echo "$(YELLOW)Deploying production manifests to Kubernetes...$(NC)"
	@$(KUBECTL) apply -f infrastructure/kubernetes/distributed-architecture.yaml
	@echo "$(GREEN)Production deployment completed!$(NC)"

k8s-deploy-dev: ## Deploy development Kubernetes manifests
	@echo "$(YELLOW)Deploying development manifests to Kubernetes...$(NC)"
	@./scripts/k8s-deploy-dev.sh $(NAMESPACE) $(REGISTRY) $(TAG) $(K8S_NODE_PORT_BASE)
	@echo "$(GREEN)Development deployment completed!$(NC)"

k8s-health: ## Check health of all Kubernetes services
	@echo "$(YELLOW)Checking Kubernetes service health...$(NC)"
	@./scripts/k8s-health-check.sh $(NAMESPACE)

k8s-logs: ## Show logs from all services
	@echo "$(YELLOW)Showing logs from all services...$(NC)"
	@$(KUBECTL) logs -n $(NAMESPACE) -l app.kubernetes.io/part-of=$(PROJECT_NAME) --tail=100 -f

k8s-logs-service: ## Show logs from specific service (usage: make k8s-logs-service SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Showing logs for $(SERVICE)...$(NC)"
	@$(KUBECTL) logs -n $(NAMESPACE) -l app=$(SERVICE) --tail=100 -f

k8s-port-forward: ## Setup port forwarding for development (usage: make k8s-port-forward SERVICE=api-gateway PORT=8000)
	@if [ -z "$(SERVICE)" ] || [ -z "$(PORT)" ]; then \
		echo "$(RED)Please specify SERVICE and PORT variables$(NC)"; \
		echo "Example: make k8s-port-forward SERVICE=api-gateway PORT=8000"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Setting up port forwarding for $(SERVICE) on port $(PORT)...$(NC)"
	@$(KUBECTL) port-forward -n $(NAMESPACE) service/$(SERVICE) $(PORT):$(PORT)

k8s-shell: ## Open shell in service pod (usage: make k8s-shell SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Opening shell in $(SERVICE) pod...$(NC)"
	@$(KUBECTL) exec -it -n $(NAMESPACE) deployment/$(SERVICE) -- /bin/bash

k8s-stop: ## Stop the Kubernetes development environment (removes namespace only)
	@echo "$(YELLOW)Stopping Kubernetes development environment...$(NC)"
	@echo "$(RED)This will delete namespace $(NAMESPACE) and all its resources! Continue? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(KUBECTL) delete namespace $(NAMESPACE) --ignore-not-found=true; \
		echo "$(GREEN)Development environment stopped!$(NC)"; \
	else \
		echo "$(YELLOW)Stop cancelled$(NC)"; \
	fi

k8s-clean: ## Clean up Kubernetes cluster and Docker resources
	@echo "$(YELLOW)Cleaning up Kubernetes cluster and Docker resources...$(NC)"
	@echo "$(RED)This will delete the entire cluster, registry, and Docker images! Continue? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		./scripts/k8s-cleanup.sh $(K8S_CLUSTER_TYPE); \
		echo "$(GREEN)Cluster and resources cleaned up!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled$(NC)"; \
	fi

k8s-clean-all: ## Force clean up everything (cluster, Docker, networks, images)
	@echo "$(RED)WARNING: This will force delete EVERYTHING including all Docker resources!$(NC)"
	@echo "$(RED)This includes: cluster, registry, networks, images, and containers! Continue? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		./scripts/k8s-cleanup.sh all; \
		docker system prune -af --volumes; \
		echo "$(GREEN)Everything cleaned up!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled$(NC)"; \
	fi

k8s-cleanup: ## Cleanup all Kubernetes resources (legacy command - use k8s-stop instead)
	@echo "$(YELLOW)Note: This is a legacy command. Use 'make k8s-stop' or 'make k8s-clean' instead.$(NC)"
	@$(MAKE) k8s-stop

k8s-restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	@$(KUBECTL) rollout restart -n $(NAMESPACE) deployment
	@echo "$(GREEN)All services restarted!$(NC)"

k8s-restart-service: ## Restart specific service (usage: make k8s-restart-service SERVICE=api-gateway)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restarting $(SERVICE)...$(NC)"
	@$(KUBECTL) rollout restart -n $(NAMESPACE) deployment/$(SERVICE)
	@echo "$(GREEN)$(SERVICE) restarted!$(NC)"

k8s-status: ## Show status of all Kubernetes resources
	@echo "$(YELLOW)Showing Kubernetes resource status...$(NC)"
	@echo "$(CYAN)Namespace:$(NC)"
	@$(KUBECTL) get namespace $(NAMESPACE)
	@echo ""
	@echo "$(CYAN)Pods:$(NC)"
	@$(KUBECTL) get pods -n $(NAMESPACE)
	@echo ""
	@echo "$(CYAN)Services:$(NC)"
	@$(KUBECTL) get services -n $(NAMESPACE)
	@echo ""
	@echo "$(CYAN)Deployments:$(NC)"
	@$(KUBECTL) get deployments -n $(NAMESPACE)

# Database Operations (Kubernetes)
k8s-db-init: ## Initialize database with migrations
	@echo "$(YELLOW)Initializing database...$(NC)"
	@$(KUBECTL) exec -n $(NAMESPACE) deployment/user-management -- alembic upgrade head
	@echo "$(GREEN)Database initialized successfully!$(NC)"

k8s-db-migrate: ## Create new database migration
	@echo "$(YELLOW)Creating new migration...$(NC)"
	@$(KUBECTL) exec -n $(NAMESPACE) deployment/user-management -- alembic revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)Migration created successfully!$(NC)"

k8s-db-upgrade: ## Apply database migrations
	@echo "$(YELLOW)Applying database migrations...$(NC)"
	@$(KUBECTL) exec -n $(NAMESPACE) deployment/user-management -- alembic upgrade head
	@echo "$(GREEN)Migrations applied successfully!$(NC)"

k8s-db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "$(RED)WARNING: This will delete all data! Are you sure? (y/N)$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(KUBECTL) delete pod -n $(NAMESPACE) -l app=postgres-primary; \
		$(KUBECTL) delete pvc -n $(NAMESPACE) -l app=postgres-primary; \
		sleep 30; \
		$(KUBECTL) wait --for=condition=ready pod -n $(NAMESPACE) -l app=postgres-primary --timeout=300s; \
		$(KUBECTL) exec -n $(NAMESPACE) deployment/user-management -- alembic upgrade head; \
		echo "$(GREEN)Database reset successfully!$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

# =======================================
# Legacy Docker Operations (for compatibility)
# =======================================

build: k8s-build ## Build all Docker images (Kubernetes version)

up: k8s-deploy-dev ## Start all services (Kubernetes version)

down: k8s-stop ## Stop all services (Kubernetes version)

restart: k8s-restart ## Restart all services (Kubernetes version)

restart-service: k8s-restart-service ## Restart a specific service (Kubernetes version)

# =======================================
# Legacy Logging and Monitoring (redirected to Kubernetes)
# =======================================

logs: k8s-logs ## Show logs from all services (Kubernetes version)

logs-service: k8s-logs-service ## Show logs from specific service (Kubernetes version)

health: k8s-health ## Check health of all services (Kubernetes version)

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

ps: k8s-status ## Show running containers (Kubernetes version)

exec: k8s-shell ## Execute shell in service container (Kubernetes version)

shell: k8s-shell ## Open bash shell in service container (Kubernetes version)

# =======================================
# Development Workflow (Kubernetes Native)
# =======================================

dev: k8s-dev ## Complete development setup (Kubernetes version)

# Legacy development workflow (redirected to Kubernetes)
setup-env-dev: setup-env k8s-dev ## Complete development setup with environment

# =======================================
# Production Commands (Kubernetes)
# =======================================

prod-build: k8s-build ## Build production images (Kubernetes version)

prod-deploy: k8s-deploy ## Deploy to production Kubernetes

prod-up: k8s-deploy ## Start production services (Kubernetes version)
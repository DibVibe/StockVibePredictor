# StockVibePredictor Makefile
# Project automation and management

# Variables
PYTHON := python3
PIP := pip3
NPM := npm
BACKEND_DIR := Backend
FRONTEND_DIR := Frontend
MANAGE := $(PYTHON) manage.py

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

.PHONY: all help install install-backend install-frontend \
        dev dev-backend dev-frontend \
        migrate makemigrations test test-backend test-frontend \
        clean clean-backend clean-frontend \
        build docker-up docker-down \
        logs lint format check

# Default target
all: help

# Help command
help:
	@echo ""
	@echo "$(GREEN)StockVibePredictor - Available Commands:$(NC)"
	@echo ""
	@echo "$(YELLOW)Installation:$(NC)"
	@echo "  make install          - Install all dependencies (backend + frontend)"
	@echo "  make install-backend  - Install Python dependencies only"
	@echo "  make install-frontend - Install Node dependencies only"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make dev             - Run both backend and frontend servers"
	@echo "  make dev-backend     - Run Django server only"
	@echo "  make dev-frontend    - Run React development server only"
	@echo ""
	@echo "$(YELLOW)Database:$(NC)"
	@echo "  make migrate         - Apply database migrations"
	@echo "  make makemigrations  - Create new migrations"
	@echo "  make superuser       - Create Django superuser"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test            - Run all tests"
	@echo "  make test-backend    - Run Django tests"
	@echo "  make test-frontend   - Run frontend tests"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-up       - Start services with Docker Compose"
	@echo "  make docker-down     - Stop Docker services"
	@echo "  make docker-build    - Build Docker images"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make clean           - Remove all temporary files"
	@echo "  make lint            - Run linters"
	@echo "  make format          - Format code"
	@echo "  make logs            - Show application logs"
	@echo "  make check           - Run all checks before commit"
	@echo ""

# Install all dependencies
install: install-backend install-frontend
	@echo "$(GREEN)✓ All dependencies installed successfully!$(NC)"

# Install backend dependencies
install-backend:
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	@cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Backend dependencies installed$(NC)"

# Install frontend dependencies
install-frontend:
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) install
	@echo "$(GREEN)✓ Frontend dependencies installed$(NC)"

# Run both servers in development mode
dev:
	@echo "$(GREEN)Starting StockVibePredictor in development mode...$(NC)"
	@echo "$(YELLOW)Backend will run on http://localhost:8000$(NC)"
	@echo "$(YELLOW)Frontend will run on http://localhost:3000$(NC)"
	@make -j 2 dev-backend dev-frontend

# Run Django development server
dev-backend:
	@echo "$(YELLOW)Starting Django server...$(NC)"
	@cd $(BACKEND_DIR) && $(MANAGE) runserver

# Run React development server
dev-frontend:
	@echo "$(YELLOW)Starting React development server...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) start

# Database migrations
migrate:
	@echo "$(YELLOW)Applying migrations...$(NC)"
	@cd $(BACKEND_DIR) && $(MANAGE) migrate
	@echo "$(GREEN)✓ Migrations applied$(NC)"

makemigrations:
	@echo "$(YELLOW)Creating migrations...$(NC)"
	@cd $(BACKEND_DIR) && $(MANAGE) makemigrations
	@echo "$(GREEN)✓ Migrations created$(NC)"

# Create superuser
superuser:
	@echo "$(YELLOW)Creating Django superuser...$(NC)"
	@cd $(BACKEND_DIR) && $(MANAGE) createsuperuser

# Run tests
test: test-backend test-frontend
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-backend:
	@echo "$(YELLOW)Running Django tests...$(NC)"
	@cd $(BACKEND_DIR) && $(MANAGE) test

test-frontend:
	@echo "$(YELLOW)Running frontend tests...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) test

# Docker commands
docker-up:
	@echo "$(YELLOW)Starting Docker services...$(NC)"
	@cd $(BACKEND_DIR) && docker-compose up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"

docker-down:
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	@cd $(BACKEND_DIR) && docker-compose down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

docker-build:
	@echo "$(YELLOW)Building Docker images...$(NC)"
	@cd $(BACKEND_DIR) && docker-compose build
	@echo "$(GREEN)✓ Docker images built$(NC)"

# Clean commands
clean: clean-backend clean-frontend
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-backend:
	@echo "$(YELLOW)Cleaning backend...$(NC)"
	@find $(BACKEND_DIR) -type f -name "*.pyc" -delete
	@find $(BACKEND_DIR) -type d -name "__pycache__" -delete
	@find $(BACKEND_DIR) -type f -name "*.pyo" -delete
	@find $(BACKEND_DIR) -type f -name ".coverage" -delete
	@find $(BACKEND_DIR) -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Backend cleaned$(NC)"

clean-frontend:
	@echo "$(YELLOW)Cleaning frontend...$(NC)"
	@cd $(FRONTEND_DIR) && rm -rf node_modules build dist
	@echo "$(GREEN)✓ Frontend cleaned$(NC)"

# Linting
lint:
	@echo "$(YELLOW)Running linters...$(NC)"
	@cd $(BACKEND_DIR) && python -m flake8 . --exclude=migrations,venv
	@cd $(FRONTEND_DIR) && $(NPM) run lint 2>/dev/null || echo "No lint script found"
	@echo "$(GREEN)✓ Linting completed$(NC)"

# Format code
format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	@cd $(BACKEND_DIR) && python -m black . --exclude=migrations
	@cd $(FRONTEND_DIR) && $(NPM) run format 2>/dev/null || echo "No format script found"
	@echo "$(GREEN)✓ Formatting completed$(NC)"

# Show logs
logs:
	@echo "$(YELLOW)Showing recent logs...$(NC)"
	@tail -f $(BACKEND_DIR)/Logs/*.log

# Run all checks before commit
check: lint test
	@echo "$(GREEN)✓ All checks passed! Ready to commit.$(NC)"

# Quick setup for new developers
setup: install migrate
	@echo "$(GREEN)✓ Project setup completed!$(NC)"
	@echo "Run 'make dev' to start the development servers"

# Production build
build:
	@echo "$(YELLOW)Building for production...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) run build
	@cd $(BACKEND_DIR) && $(PYTHON) manage.py collectstatic --noinput
	@echo "$(GREEN)✓ Production build completed$(NC)"

# Run with specific ports (optional)
dev-custom:
	@echo "$(GREEN)Starting with custom ports...$(NC)"
	@make -j 2 dev-backend-custom dev-frontend-custom

dev-backend-custom:
	@cd $(BACKEND_DIR) && $(MANAGE) runserver 0.0.0.0:8080

dev-frontend-custom:
	@cd $(FRONTEND_DIR) && PORT=3001 $(NPM) start

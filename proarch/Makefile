.PHONY: help install dev-install lint format test test-unit test-integration clean docker-build docker-run ci-local

# Default target
help:
	@echo "Available commands:"
	@echo "  install       - Install production dependencies"
	@echo "  dev-install   - Install development dependencies"
	@echo "  lint          - Run linting (flake8)"
	@echo "  format        - Format code with black"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  clean         - Clean cache files"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  ci-local      - Run CI checks locally"

# Installation
install:
	pip install -r requirements.txt

dev-install: install
	pip install -r requirements-dev.txt
	pre-commit install

# Code Quality
lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

# Testing
test:
	python test_runner.py

test-unit:
	python -m pytest tests/test_etl.py tests/test_kpi.py tests/test_retriever.py tests/test_api.py -v

test-integration:
	python -m pytest tests/ -k "integration" -v

# Coverage
coverage:
	pip install coverage
	coverage run -m pytest tests/ -v
	coverage report
	coverage html

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

# Docker
docker-build:
	docker build -t proarch-analytics .

docker-build-api:
	docker build -f Dockerfile.api -t proarch-api .

docker-build-frontend:
	docker build -f Dockerfile.frontend -t proarch-frontend .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 proarch-analytics

docker-compose-up:
	docker-compose up

docker-compose-up-d:
	docker-compose up -d

docker-compose-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

docker-compose-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

docker-compose-init:
	docker-compose --profile init run --rm init-data

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

docker-clean:
	docker system prune -f
	docker image prune -f

# CI Simulation
ci-local: lint format type-check test

# Development server
run-api:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run frontend.py

run-ingestion:
	python run_ingestion_fixed.py

# Database
init-db:
	python -c "from src.ingestion.schema import init_db; init_db('sqlite:///./data/cost_analytics.db')"

check-db:
	python check_db.py

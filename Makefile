# XAUUSD Market Structure Detection - Makefile
# Development and deployment commands

.PHONY: help setup install install-dev test test-coverage lint format clean docker-build docker-run docs

# Default target
help:
	@echo "Available commands:"
	@echo "  setup          - Initial project setup"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  test           - Run unit tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean temporary files"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docs           - Generate documentation"

# Project setup
setup:
	@echo "Setting up XAUUSD Market Structure Detection project..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -c "from src.utils.helpers import setup_project_directories; setup_project_directories()"
	@echo "Setup complete!"

# Install dependencies
install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest-xdist pytest-mock

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "Coverage report generated in htmlcov/"

test-watch:
	python -m pytest tests/ -v --tb=short -x --looponfail

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=_pycache_
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black --line-length=100

format-check:
	black src/ tests/ --line-length=100 --check
	isort src/ tests/ --profile black --line-length=100 --check-only

# Data processing commands
load-sample-data:
	python -c "from src.utils.helpers import create_sample_data; df = create_sample_data(1000); df.to_csv('data/raw/sample_xauusd_15m.csv', index=False); print('Sample data created')"

process-data:
	python -c "from src.data.loader import DataLoader; from src.data.resample import DataResampler; loader = DataLoader(); resampler = DataResampler(); df = loader.load_csv('data/raw/sample_xauusd_15m.csv'); clean_file = loader.save_clean_data(df, 'sample_clean.csv'); resampled = resampler.resample_all_timeframes(df); resampler.save_resampled_data(resampled, 'sample'); print('Data processing complete')"

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "_pycache_" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

clean-data:
	rm -rf data/clean/*
	rm -rf data/resampled/*
	rm -rf outputs/events/*
	rm -rf logs/*

# Docker commands
docker-build:
	docker build -t xauusd-market-structure:latest .

docker-run: docker-build
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/logs:/app/logs \
		xauusd-market-structure:latest

docker-shell: docker-build
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/logs:/app/logs \
		xauusd-market-structure:latest /bin/bash

docker-test:
	docker run --rm \
		-v $(PWD):/app \
		xauusd-market-structure:latest \
		python -m pytest tests/ -v

# Development server (for future Streamlit app)
dev-server:
	streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Documentation (placeholder for future)
docs:
	@echo "Documentation generation not implemented yet"
	@echo "Future: sphinx-build -b html docs/ docs/_build/"

# CI/CD helpers
ci-setup: install-dev
	@echo "CI environment setup complete"

ci-test: format-check lint test-coverage
	@echo "All CI checks passed"

# Jupyter notebook management
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook-clean:
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

# Database operations (future)
migrate:
	@echo "Database migrations not implemented yet"

seed-db:
	@echo "Database seeding not implemented yet"

# Deployment (future)
deploy-dev:
	@echo "Development deployment not implemented yet"

deploy-prod:
	@echo "Production deployment not implemented yet"

# Performance testing (future)
benchmark:
	@echo "Performance benchmarking not implemented yet"

# Quick development workflow
dev: format lint test
	@echo "Development workflow complete"

# Full CI workflow
ci: ci-setup ci-test
	@echo "Full CI workflow complete"
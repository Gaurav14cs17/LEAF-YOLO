# LEAF-YOLO Makefile for development and testing

.PHONY: help install install-dev test test-fast test-slow test-integration test-unit test-benchmark lint format clean coverage docs

help:  ## Show this help message
	@echo "LEAF-YOLO Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install package and dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov pytest-benchmark pytest-xdist pytest-timeout
	pip install black flake8 isort mypy
	pip install sphinx sphinx-rtd-theme

# Testing
test:  ## Run all tests
	pytest tests/ -v

test-fast:  ## Run fast tests only (exclude slow and benchmark tests)
	pytest tests/ -v -m "not slow and not benchmark"

test-slow:  ## Run slow tests only
	pytest tests/ -v -m "slow"

test-integration:  ## Run integration tests
	pytest tests/integration/ -v

test-unit:  ## Run unit tests
	pytest tests/unit/ -v

test-benchmark:  ## Run performance benchmarks
	pytest tests/benchmarks/ -v -m "benchmark" --benchmark-only

test-parallel:  ## Run tests in parallel
	pytest tests/ -v -n auto

test-gpu:  ## Run GPU-specific tests (if CUDA available)
	pytest tests/ -v -m "gpu"

# Code Quality
lint:  ## Run linting checks
	flake8 leafyolo tests
	mypy leafyolo --ignore-missing-imports

format:  ## Format code with black and isort
	black leafyolo tests
	isort leafyolo tests

format-check:  ## Check code formatting
	black --check leafyolo tests
	isort --check leafyolo tests

# Coverage
coverage:  ## Run tests with coverage report
	pytest tests/ --cov=leafyolo --cov-report=html --cov-report=term

coverage-xml:  ## Generate XML coverage report for CI
	pytest tests/ --cov=leafyolo --cov-report=xml

# Documentation
docs:  ## Build documentation
	cd docs && make html

docs-clean:  ## Clean documentation build
	cd docs && make clean

# Cleanup
clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development
dev-setup:  ## Set up development environment
	make install-dev
	pre-commit install

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

# Docker
docker-build:  ## Build Docker image
	docker build -t leafyolo:latest .

docker-test:  ## Run tests in Docker
	docker run --rm -v $(PWD):/workspace leafyolo:latest make test-fast

# Model Testing
test-models:  ## Test model inference with different variants
	python -c "
from leafyolo import LEAFYOLO
variants = ['leafyolo_n', 'leafyolo_s', 'leafyolo_m']
for variant in variants:
    try:
        model = LEAFYOLO('detect', variant=variant)
        print(f'✅ {variant}: Model created successfully')
    except Exception as e:
        print(f'❌ {variant}: {e}')
"

test-config:  ## Test configuration system
	python -c "
from leafyolo.utils.config import get_config
tasks = ['detect', 'segment', 'classify']
variants = ['leafyolo_n', 'leafyolo_s', 'leafyolo_m']
for task in tasks:
    for variant in variants:
        try:
            config = get_config(task, variant)
            print(f'✅ {task}/{variant}: Config generated successfully')
        except Exception as e:
            print(f'❌ {task}/{variant}: {e}')
"

# Continuous Integration
ci-test:  ## Run CI test suite
	make test-fast
	make test-integration
	make lint
	make coverage-xml

# Performance Testing
perf-test:  ## Run performance benchmarks with profiling
	pytest tests/benchmarks/ -v --benchmark-only --benchmark-json=benchmark_results.json

perf-compare:  ## Compare performance against baseline
	pytest-benchmark compare baseline_benchmark.json benchmark_results.json

# Release
check-release:  ## Check if package is ready for release
	make lint
	make test
	python setup.py check --strict

build:  ## Build distribution packages
	python setup.py sdist bdist_wheel

# Utilities
count-lines:  ## Count lines of code
	find leafyolo -name "*.py" | xargs wc -l | tail -1
	find tests -name "*.py" | xargs wc -l | tail -1

check-deps:  ## Check for outdated dependencies
	pip list --outdated

update-deps:  ## Update dependencies to latest versions
	pip freeze | cut -d= -f1 | xargs pip install -U

# Examples
run-example-train:  ## Run example training
	python train_leafyolo.py --task detect --variant leafyolo_n --data coco --epochs 1 --batch-size 2

run-example-predict:  ## Run example prediction
	python predict_leafyolo.py --model cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --source visualize_img/ --save

run-example-export:  ## Run example export
	python leafyolo_cli.py export --model cfg/LEAF-YOLO/leaf-sizes/weights/best.pt --format onnx

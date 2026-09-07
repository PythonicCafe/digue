PYTHON = python3

help:					# List all make commands
	@awk -F ':.*#' '/^[a-zA-Z_-]+:.*?#/ { printf "\033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort

dev-install:				# Install dev dependencies (pytest, mypy, ruff, build, twine)
	$(PYTHON) -m pip install --user --group dev .

docker-build:				# Build the docker container for local development
	docker build --build-arg ENV_TYPE=development -t turicas/digue:latest -f Dockerfile .

test:					# Execute pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage:				# Execute pytest with coverage report
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=digue --cov-report=term-missing

test-fast:				# Execute pytest quietly (no verbose, last failures only)
	$(PYTHON) -m pytest tests/ -q --tb=line -x --lf

mypy:					# Execute `mypy --strict` in the codebase
	$(PYTHON) -m mypy

lint:					# Run ruff (check --fix + format)
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format --line-length 120 .

lint-check:				# Run ruff without changing files
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --line-length 120 --check .

check: lint-check mypy test		# Run everything (lint-check, mypy, test)

build:					# Build sdist and wheel into clean build and dist directories
	rm -rf build/ dist/
	$(PYTHON) -m build

smoke-wheel: build			# Install wheel without dependencies and smoke-test its CLI
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	$(PYTHON) -m venv "$$tmp_dir/venv"; \
	"$$tmp_dir/venv/bin/python" -m pip install --no-deps dist/*.whl; \
	"$$tmp_dir/venv/bin/digue" --version; \
	"$$tmp_dir/venv/bin/digue" config show >/dev/null

smoke-sdist: build			# Run the test suite from the published sdist (needs tests/ + conftest.py in it)
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	tar xzf dist/*.tar.gz -C "$$tmp_dir"; \
	cd "$$tmp_dir"/digue-*/ && $(PYTHON) -m pytest tests/ -q --tb=short

build-check: smoke-wheel smoke-sdist	# Build, smoke-test (wheel CLI + sdist suite), and validate with twine
	$(PYTHON) -m twine check dist/*

publish: build-check			# Upload packages to PyPI (requires credentials)
	$(PYTHON) -m twine upload dist/*

clean:					# Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

tags:					# Generate tags file for the entire project (requires universal-ctags)
	@git ls-files | ctags -L - --tag-relative=yes --quiet --append -f ".tags"

.PHONY: help dev-install docker-build test test-coverage test-fast mypy lint lint-check check build smoke-wheel build-check publish clean tags

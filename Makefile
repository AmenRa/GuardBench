.DEFAULT_GOAL := help

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHONPATH=
VENV = .venv
MAKE = make

VENV_BIN=$(VENV)/bin

.venv:  ## Set up virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh all project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements-dev.txt
	$(VENV_BIN)/pip install -r requirements.txt

.PHONY: build
build: .venv  ## Compile and install GuardBench
	. $(VENV_BIN)/activate

.PHONY: fix-lint
fix-lint: .venv  ## Fix linting
	. $(VENV_BIN)/activate
	$(VENV_BIN)/black ./guardbench
	$(VENV_BIN)/isort ./guardbench

.PHONY: lint
lint: .venv  ## Check linting
	. $(VENV_BIN)/activate
	$(VENV_BIN)/isort --check ./guardbench
	$(VENV_BIN)/black --check ./guardbench
	$(VENV_BIN)/blackdoc ./guardbench
	$(VENV_BIN)/ruff check ./guardbench
	$(VENV_BIN)/typos ./guardbench
#	$(VENV_BIN)/mypy guardbench

.PHONY: test
test: .venv build  ## Run unittest
	$(VENV_BIN)/pytest

.PHONY: coverage
coverage: .venv build  ## Run tests and report coverage
	$(VENV_BIN)/pytest --cov -n auto --dist worksteal -m "not benchmark"

.PHONY: sbom
sbom: .venv build  ## Generate sbom
	$(VENV_BIN)/pip-licenses --with-authors > sbom.txt

.PHONY: sast
sast: .venv build  ## Run SAST using Bandit https://github.com/PyCQA/bandit
	$(VENV_BIN)/bandit -r ./guardbench > sast.log
	$(VENV_BIN)/bandit -r ./scripts > sast-scripts.log

.PHONY: release
release: .venv build  ## Release a new version
	$(VENV_BIN)/python setup.py sdist bdist_wheel
	$(VENV_BIN)/twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: download-all
download-all: .venv build  ## Download all GuardBench's datasets
	$(VENV_BIN)/python -c "import guardbench; guardbench.download_all()"

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf target/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf guardbench.egg-info
	@rm -rf dist
	@rm -rf build
	@rm -rf sbom
	@rm -rf sast.log
	@rm -rf sast-scripts.log
	
.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
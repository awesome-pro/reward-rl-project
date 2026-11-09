PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup prepare audit experiment all test clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

prepare:
	$(PYTHON) -m rewardforge.cli prepare

audit:
	$(PYTHON) -m rewardforge.cli audit

experiment:
	$(PYTHON) -m rewardforge.cli experiment

all:
	$(PYTHON) -m rewardforge.cli all

test:
	$(PYTHON) -m pytest

clean:
	rm -rf artifacts data/processed data/private data/raw .pytest_cache

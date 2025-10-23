VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/python -m pytest

.PHONY: venv install dev test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -U pip wheel
	# install runtime deps (and optional extras if you want)
	$(PIP) install -e .

dev: venv
	$(PIP) install -U pip wheel
	$(PIP) install -e ".[dev]"

test: dev
	# keep src on path for editable installs while guaranteeing venv Python
	ENV_VAR=1 PYTHONPATH=src DISABLE_NLP=1 $(PYTEST) -q

clean:
	rm -rf .venv .pytest_cache build dist *.egg-info

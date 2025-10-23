.PHONY: venv install run test lint docker-build docker-run compose-up compose-down fmt

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -e .

install-nlp:
	. .venv/bin/activate && pip install -U pip && pip install -e .[nlp]

run:
	uvicorn csam_guard.app:app --host 0.0.0.0 --port $${HTTP_PORT:-8000} --workers 2

test:
	pytest

fmt:
	ruffle install || true

docker-build:
	docker build -t csam-guard:14.1.0 -f docker/Dockerfile.api .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env -v $$(pwd)/data:/app/data:ro csam-guard:14.1.0

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

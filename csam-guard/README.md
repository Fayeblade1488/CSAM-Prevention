# CSAM Guard

A production-oriented API to **detect and prevent** child-safety risk signals in text and images. It uses layered heuristics (normalization, fuzzy/phonetic match, SimHash, context clusters), optional NLP scoring, and image pHash matching against known bad hashes. **No raw content is persisted**—only SHA-256 hashes and minimal signals.

> This project is for **prevention** and moderation. It will never generate content.

## Features
- FastAPI service with `/assess` (text) and `/assess_image` (image) endpoints
- Normalization, obfuscation handling, age parsing (numeric & spelled)
- Hard/ambiguous term detection, cross-sentence checks, cluster heuristics
- Professional context allowlisting and adult-assertion validation
- Image perceptual hash matching with configurable Hamming threshold
- Optional RSS-based term proposals (manual endpoint call)
- Optional NLP model gating via `NLP_ENABLED=1`
- Prometheus metrics and Grafana dashboard
- Docker, Compose, Kubernetes (and Helm chart)
- Unit tests

## Quick Start
```bash
cp .env.sample .env
docker compose up --build
# API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Local (without Docker)
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
uvicorn csam_guard.app:app --port 8000 --host 0.0.0.0
```

### Optional NLP
The NLP classifier is **disabled by default**. To enable:
```bash
pip install -e .[nlp]  # and install a backend like torch or tensorflow
export NLP_ENABLED=1
```
Then restart the service.

## Security & Privacy
- Logs hold **only SHA-256 hashes** of inputs (never the raw text or image bytes).
- Rate limiting and upload size caps are enforced.
- When behind a proxy, set `TRUST_XFF=1` to rate-limit on `X-Forwarded-For`.

## Configuration (env)
See `.env.sample` for commonly tuned settings.

## Kubernetes
Simplified manifests are in `k8s/`. For production, use the Helm chart in `charts/` with your values.

## Known Hashes
Populate `data/known_hashes.json` with hex-encoded 64-bit pHashes and rotate regularly. Treat it as sensitive and mount as a secret in production.

## License
MIT

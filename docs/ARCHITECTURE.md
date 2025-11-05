# Architecture

- **API**: FastAPI + Uvicorn
- **Core**: `CSAMGuard`—text analyzer and image pHash matcher
- **Metrics**: Prometheus counters (requests & decisions), Grafana dashboard
- **Deploy**: Docker/Compose, Kubernetes, Helm
- **Privacy**: no raw content at rest; logs are hashes only

RSS updates and NLP scoring are optional paths behind admin endpoints or env flags.

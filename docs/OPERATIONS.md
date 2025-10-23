## Running locally
To run the service locally for development, use the following commands:
```bash
make dev
make run
```
The service will be available at `http://localhost:8000`.

## Deploy
The service is deployed as a Docker container. The `release.yml` workflow in `.github/workflows/` builds and publishes a new release when a new tag is pushed to the repository.

## Monitoring
The service exposes Prometheus metrics at the `/metrics` endpoint. These metrics can be scraped by a Prometheus server and visualized in Grafana.

## Backups / Data retention
The service does not store any data, so there are no backup or data retention policies.

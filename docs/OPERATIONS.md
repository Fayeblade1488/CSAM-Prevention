# Operations

This document provides guidance on deploying, monitoring, and maintaining the CSAM-Prevention service.

## Deployment
The application can be deployed using Docker or a standard Python web server like Uvicorn. The `Makefile` provides convenient targets for building and running the service.

## Monitoring
The service exposes a `/metrics` endpoint for Prometheus monitoring. This endpoint provides metrics on the number of requests, the number of blocks, and other useful information.

The service also includes a `/health` endpoint that can be used for health checks.

## Configuration
The service can be configured using environment variables. The following variables are available:

- `PROMETHEUS_ENABLED`: Set to `1` to enable Prometheus metrics.
- `MAX_UPLOAD_BYTES`: The maximum allowed size for uploaded images.
- `HTTP_PORT`: The port to run the service on.
- `TRUST_XFF`: Set to `1` to trust the `X-Forwarded-For` header.
- `HASH_LIST_PATH`: The path to a file containing a list of known CSAM image hashes.

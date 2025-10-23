## Overview
The CSAM Guard service is a FastAPI application that exposes two main endpoints: `/assess` for text analysis and `/assess_image` for image analysis. The core logic is encapsulated in the `CSAMGuard` class, which is responsible for processing the input and making a decision based on a set of configurable rules.

## Key components
- **`app.py`**: The FastAPI application that defines the API endpoints and handles incoming requests.
- **`guard.py`**: The core of the service, containing the `CSAMGuard` class and all the logic for text and image analysis.
- **`rules/RULES.json`**: A machine-readable file containing the ruleset used for detection.

## Data & Config
- **Configuration**: The service is configured through environment variables and the `rules/RULES.json` file.
- **Secrets**: The service does not require any secrets to be stored in the repository.

## Observability
- **Logging**: The service uses the standard Python `logging` module to log information about its operations.
- **Metrics**: The service exposes Prometheus metrics at the `/metrics` endpoint.

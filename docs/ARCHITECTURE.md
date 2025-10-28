# Architecture

The application is a FastAPI service with two main components:

- **`csam_guard.guard.CSAMGuard`**: This class contains the core logic for assessing text and images. It uses a combination of heuristics, fuzzy matching, and an optional NLP model to detect potential risks.
- **`csam_guard.app`**: This module exposes the `CSAMGuard` functionality through a REST API. It handles incoming requests, manages the application lifecycle, and provides endpoints for content assessment, health checks, and version information.

The service is designed to be stateless, making it easy to deploy and scale horizontally. It can be run as a standalone service or integrated into a larger microservices architecture.

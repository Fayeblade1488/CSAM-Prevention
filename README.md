# CSAM Guard

CSAM Guard is a production-oriented API designed to detect and prevent child-safety risk signals in text and images. It employs a multi-layered approach that includes heuristics, optional NLP scoring, and image perceptual hash matching to identify potentially harmful content. The system is built with privacy and security in mind, ensuring that no raw content is ever persisted.

> This project is for **prevention** and moderation. It will never generate content.

## Features

-   **FastAPI Service**: Provides `/assess` (text) and `/assess_image` (image) endpoints for content analysis.
-   **Advanced Text Analysis**: Utilizes normalization, obfuscation handling, and age parsing (both numeric and spelled out) to understand the context of the text.
-   **Heuristic-Based Detection**: Implements hard and ambiguous term detection, cross-sentence checks, and cluster heuristics to identify risky content.
-   **Context-Aware Filtering**: Includes a professional context allowlist and adult-assertion validation to reduce false positives.
-   **Image Hashing**: Matches image perceptual hashes against a database of known harmful content with a configurable Hamming threshold.
-   **Dynamic Term Updates**: Supports optional RSS-based term proposals to keep the detection system up-to-date.
-   **Optional NLP Scoring**: Can be enabled with `NLP_ENABLED=1` to provide an additional layer of analysis.
-   **Monitoring and Deployment**: Comes with Prometheus metrics, a Grafana dashboard, and support for Docker, Docker Compose, Kubernetes, and Helm.
-   **Comprehensive Testing**: Includes a suite of unit tests to ensure reliability.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Docker and Docker Compose (for containerized deployment)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/csam-guard.git
    cd csam-guard
    ```

2.  **Set up the environment**:
    ```bash
    cp .env.sample .env
    ```

3.  **Run with Docker Compose**:
    ```bash
    docker compose up --build
    ```

    The API will be available at `http://localhost:8000/docs`, Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000` (login with `admin/admin`).

### Local Development (without Docker)

1.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -e .
    ```

3.  **Run the application**:
    ```bash
    uvicorn csam_guard.app:app --port 8000 --host 0.0.0.0
    ```

### Enabling the NLP Classifier

The NLP classifier is disabled by default. To enable it:

1.  **Install the NLP dependencies**:
    ```bash
    pip install -e .[nlp]
    ```

2.  **Set the environment variable**:
    ```bash
    export NLP_ENABLED=1
    ```

3.  **Restart the service**.

## API Usage

### Text Assessment

To assess a text prompt, send a POST request to the `/assess` endpoint:

```bash
curl -X POST "http://localhost:8000/assess" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "This is a test prompt."}'
```

### Image Assessment

To assess an image, send a POST request to the `/assess_image` endpoint with the image file:

```bash
curl -X POST "http://localhost:8000/assess_image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

## Security and Privacy

-   **Data Privacy**: The system is designed to never store raw text or image bytes. Logs only contain SHA-256 hashes of the inputs.
-   **Rate Limiting**: The API enforces rate limiting and upload size caps to prevent abuse.
-   **Proxy Support**: If the service is behind a proxy, set `TRUST_XFF=1` to enable rate limiting based on the `X-Forwarded-For` header.

## Configuration

For a complete list of configurable settings, see the `.env.sample` file.

## Deployment

### Kubernetes

Simplified Kubernetes manifests are available in the `k8s/` directory. For production deployments, it is recommended to use the Helm chart located in the `charts/` directory.

### Known Hashes

The `data/known_hashes.json` file should be populated with hex-encoded 64-bit perceptual hashes of known CSAM content. This file should be treated as sensitive and mounted as a secret in production environments.

## License

This project is licensed under the MIT License.

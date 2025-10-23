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

-   Python 3.10, 3.11, or 3.12 (3.11+ recommended)
-   Docker and Docker Compose (optional, for containerized deployment)
-   pip and virtualenv (for local development)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Fayeblade1488/CSAM-Prevention.git
    cd CSAM-Prevention
    ```

2.  **Set up the environment** (if you have a .env.sample file):
    ```bash
    cp .env.sample .env
    # Edit .env to configure your settings
    ```

3.  **Run with Docker Compose** (recommended for production):
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

The system can be configured through environment variables:

### Core Settings

-   `DISABLE_NLP`: Set to `"1"` to disable NLP classification and use heuristics only (default: `"0"`)
-   `PROMETHEUS_ENABLED`: Set to `"1"` to enable Prometheus metrics (default: `"0"`)
-   `HTTP_PORT`: Port for the HTTP server (default: `8000`)
-   `MAX_UPLOAD_BYTES`: Maximum upload size for images (default: `10000000` bytes)
-   `TRUST_XFF`: Set to `"1"` to trust X-Forwarded-For headers for rate limiting (default: `"0"`)
-   `HASH_LIST_PATH`: Path to a JSON file containing known CSAM perceptual hashes

### Advanced Configuration

The `DEFAULT_CONFIG` in `src/csam_guard/guard.py` contains many configuration options including:

-   **Term lists**: `hard_terms`, `ambiguous_youth`, `adult_assertions`
-   **Thresholds**: `context_threshold`, `nlp_threshold`, `phash_match_thresh`
-   **Rate limiting**: `rate_limit_max`, `rate_limit_window`
-   **RSS feeds**: List of URLs for term updates
-   **NLP model**: `nlp_model_name` and `nlp_model_version`

For a complete list of configurable settings, consult the `DEFAULT_CONFIG` dictionary in the source code.

## Deployment

### Kubernetes

Simplified Kubernetes manifests are available in the `k8s/` directory. For production deployments, it is recommended to use the Helm chart located in the `charts/` directory.

### Known Hashes

The `data/known_hashes.json` file should be populated with hex-encoded 64-bit perceptual hashes of known CSAM content. This file should be treated as sensitive and mounted as a secret in production environments.

## Testing

The project includes a comprehensive test suite with over 62 tests covering:

-   Text assessment with various edge cases
-   Image assessment and perceptual hashing
-   Rate limiting with thread-safety tests
-   API endpoints
-   Configuration validation
-   Bug regression tests
-   Concurrent access scenarios

To run the tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=csam_guard --cov-report=term

# Run specific test file
pytest tests/test_text.py -v

# Run bug fix regression tests
pytest tests/test_bug_fixes.py -v
```

Current test coverage: **82%** (650 statements, 116 missed)

### Test Categories:
- **Unit Tests**: Core functionality testing (`test_guard_extended.py`)
- **Integration Tests**: API endpoint testing (`test_api.py`)
- **Regression Tests**: Bug fix verification (`test_bug_fixes.py`, `test_bug_*.py`)
- **Edge Case Tests**: Boundary conditions and error handling

## Contributing

Contributions are welcome! Please ensure:

1.  All tests pass: `pytest`
2.  Code is linted: `ruff check src/ tests/`
3.  Code is formatted: `ruff format src/ tests/`
4.  Type checking passes: `mypy src/`
5.  Test coverage is maintained or improved
6.  All functions have comprehensive docstrings
7.  Bug fixes include regression tests

### Recent Improvements (October 2025)

- ✅ **Comprehensive Documentation**: All functions now have detailed docstrings
- ✅ **Bug Fixes**: Fixed FastAPI deprecation warning and RateLimiter race condition
- ✅ **Thread Safety**: RateLimiter now properly handles concurrent requests
- ✅ **Memory Leak Prevention**: Automatic cleanup of old rate limiter entries
- ✅ **Zero Linting Errors**: All code quality issues resolved
- ✅ **Security Verified**: CodeQL scan shows 0 vulnerabilities
- ✅ **7 New Tests**: Comprehensive regression test coverage

See `docs/COMPREHENSIVE_IMPROVEMENTS.md` and `docs/BUG_REPORT.md` for details.

## License

This project is licensed under the MIT License.

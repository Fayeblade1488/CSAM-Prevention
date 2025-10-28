# CSAM-Prevention

> A FastAPI service to detect and prevent child-safety risks in text and images.

[![CI](https://github.com/Fayeblade1488/CSAM-Prevention/workflows/ci/badge.svg)](https://github.com/Fayeblade1488/CSAM-Prevention/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](legal/LICENSE)
[![Security](https://img.shields.io/badge/security-scorecard-green.svg)](https://securityscorecards.dev/viewer/?uri=github.com/Fayeblade1488/CSAM-Prevention)

## Features

- **Text Analysis:** Detects risky keywords, phrases, and patterns in text.
- **Image Analysis:** Uses perceptual hashing to identify known CSAM images.
- **Extensible:** Easily add new detection rules and integrations.

## Quickstart

```bash
# Development setup
make setup
make test

# Production usage
make run
```

## Configuration

- **Environment variables:** `LOG_LEVEL`, `API_KEY`, `DATABASE_URL`
- **Config files:** `config/default.yaml` (schema in [docs/OPERATIONS.md](docs/OPERATIONS.md))
- **Secrets:** Never commit; use environment variables or secret managers

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and components
- [Operations](docs/OPERATIONS.md) — Deployment and monitoring
- [API Reference](docs/API.md) — Endpoints and usage *(if applicable)*
- [Contributing](CONTRIBUTING.md) — Development workflow
- [Security](SECURITY.md) — Vulnerability reporting

## License

See [legal/LICENSE](legal/LICENSE). Third-party notices in [legal/NOTICE.md](legal/NOTICE.md).

## Support

See [SUPPORT.md](SUPPORT.md) for help channels and SLAs.

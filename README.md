# CSAM Guard

A production-oriented API to detect and prevent child-safety risk signals in text and images.

## Features
- FastAPI service with `/assess` (text) and `/assess_image` (image) endpoints
- Advanced text analysis with normalization, obfuscation handling, and age parsing
- Heuristic-based detection with hard/ambiguous term matching, cross-sentence checks, and cluster analysis
- Image perceptual hash matching against a known database of harmful content
- Optional NLP scoring for an additional layer of analysis
- Dynamic term updates from RSS feeds

## Quickstart
```bash
# option A: dev
make dev
make test

# option B: run
make run
```

## Configuration
- Env vars: `PROMETHEUS_ENABLED`, `MAX_UPLOAD_BYTES`, `HTTP_PORT`, `HASH_LIST_PATH`, `TRUST_XFF`, `NLP_ENABLED`, `DISABLE_NLP`
- Files: `rules/RULES.json`

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Security
See [SECURITY.md](SECURITY.md). No secrets in code; scans run in CI.

## License
See [legal/LICENSE](legal/LICENSE).

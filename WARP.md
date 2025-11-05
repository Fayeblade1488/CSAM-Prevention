# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

CSAM-Prevention is a FastAPI service that detects and prevents child-safety risks in text and images. It uses a multi-layered detection approach combining heuristics (keyword matching, fuzzy matching, pattern recognition), optional NLP classification (transformers-based), and perceptual hashing for image analysis. The system is designed for high precision with fallback mechanisms when ML models are unavailable.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
make dev

# Install runtime dependencies only (no dev tools)
make install
```

### Testing
```bash
# Run all tests with coverage (NLP disabled for speed)
make test

# Run specific test file
DISABLE_NLP=1 .venv/bin/python -m pytest tests/test_guard_extended.py -v

# Run single test by name
DISABLE_NLP=1 .venv/bin/python -m pytest tests/test_api.py::test_assess_endpoint -v

# Run tests with NLP enabled (slower, requires transformers)
ENV_VAR=1 PYTHONPATH=src .venv/bin/python -m pytest --cov=src --cov-report=term-missing

# Run with verbose output
DISABLE_NLP=1 .venv/bin/python -m pytest -vv
```

### Linting and Type Checking
```bash
# Run both ruff and mypy
make lint

# Run ruff linter only
.venv/bin/ruff check src tests

# Run mypy type checker only
.venv/bin/mypy src

# Auto-fix ruff issues
.venv/bin/ruff check src tests --fix

# Format code
.venv/bin/ruff format src tests
```

### Running the Service
```bash
# Run locally with uvicorn (development mode)
.venv/bin/uvicorn csam_guard.app:app --reload --port 8000

# Run with Prometheus metrics enabled
PROMETHEUS_ENABLED=1 .venv/bin/uvicorn csam_guard.app:app --host 0.0.0.0 --port 8000

# Build and run Docker image
docker build -f docker/Dockerfile.api -t csam-guard:latest .
docker run -p 8000:8000 -e PROMETHEUS_ENABLED=1 csam-guard:latest
```

### Cleanup
```bash
# Remove virtual environment and build artifacts
make clean
```

## Architecture

### Core Components

**src/csam_guard/app.py** - FastAPI application layer
- Defines REST API endpoints (`/assess`, `/assess_image`, `/health`, `/version`, `/update_terms`)
- Handles request validation, rate limiting, and file upload size constraints
- Manages application lifespan (startup/shutdown) and dependency injection
- Optionally mounts Prometheus metrics endpoint at `/metrics`

**src/csam_guard/guard.py** - Detection engine
- `CSAMGuard` class implements the core detection logic
- Contains term lists, pattern matchers, fuzzy matching algorithms, and decision logic
- Handles NLP model initialization (optional, with graceful fallback)
- Implements perceptual hashing for image analysis

### Detection Pipeline

#### Text Assessment Flow (`/assess` endpoint)
1. **Signal Extraction** (`_extract_signals`)
   - Normalize text: remove zero-width chars, convert homoglyphs, apply NFKC normalization, handle leet speak
   - Apply multiple detection passes:
     - Direct term matching (hard_terms, ambiguous_youth)
     - Age extraction (numeric and spelled-out: "12 yo", "twelve years old")
     - School context detection (grade levels, school types)
     - Injection pattern detection (jailbreak attempts)
     - Cross-sentence boundary detection
     - Cluster detection (risky term combinations)

2. **Second-Pass Fuzzy Detection** (`_second_pass_detect`)
   - SimHash (locality-sensitive hashing for near-matches)
   - Jaccard similarity (character n-gram comparison)
   - Soundex (phonetic matching for misspellings)
   - Catches obfuscated variations: "k!d", "ch1ld", "l o l i"

3. **Decision Logic** (`_make_decision`)
   - Priority-based evaluation:
     1. Professional context (teacher, doctor + no sexual terms) → ALLOW
     2. Hard terms or minor age (< 18) → BLOCK (severity: CRITICAL/HIGH)
     3. Ambiguous terms with valid adult assertion → ALLOW
     4. Ambiguous terms without adult assertion → BLOCK (severity: MEDIUM/LOW)
     5. High context score or fuzzy matches → BLOCK
     6. NLP classification (if enabled and score low) → BLOCK if NSFW
     7. Default → ALLOW

4. **Context Scoring** (`_context_score`)
   - Weighted risk score calculation:
     - Hard terms: 5 points each
     - Ambiguous youth terms: 1-5 points (configurable weights)
     - School context: 2 points each
     - Injections: 6 points each
     - Minor age: 7 points
     - Second-pass matches: 3 points each
   - Reduced by 5 points for valid adult assertions
   - Threshold (default 4) determines blocking

#### Image Assessment Flow (`/assess_image` endpoint)
1. **Perceptual Hash Computation** (`_compute_phash`)
   - Convert to grayscale, resize to 32x32
   - Apply DCT (discrete cosine transform)
   - Extract 64-bit hash from low-frequency coefficients
   - Robust to minor image modifications (cropping, compression, filtering)

2. **Hash Comparison** (`assess_image`)
   - Compare against known CSAM hashes (Hamming distance)
   - Threshold (default 10 bits) determines match
   - BLOCK if distance ≤ threshold, ALLOW otherwise

### Configuration System

**DEFAULT_CONFIG** (in guard.py) - Comprehensive configuration dictionary
- **Term Lists**: hard_terms, ambiguous_youth, adult_assertions, injections
- **Patterns**: age_patterns, school_patterns, injection_patterns, allowlist_patterns
- **Thresholds**: fuzzy_threshold (0.85), hamming_thresh (5), context_threshold (4), nlp_threshold (0.7)
- **NLP Model**: nlp_model_name, nlp_model_version
- **RSS Feeds**: List of child safety feeds for term updates
- **Pending Terms**: Temporary terms with TTL (24h) and confirmation threshold (2 sources)

**Environment Variables** (app.py)
- `DISABLE_NLP=1` - Disable transformers model (faster, heuristics-only)
- `PROMETHEUS_ENABLED=1` - Enable /metrics endpoint
- `MAX_UPLOAD_BYTES` - Max image size (default 10MB)
- `HTTP_PORT` - Server port (default 8000)
- `TRUST_XFF=1` - Trust X-Forwarded-For header for rate limiting
- `HASH_LIST_PATH` - JSON file with known_csam_phashes array

### Key Implementation Details

#### NLP Model Loading (Lazy and Optional)
- Model loading happens at startup in `_load_nlp_model`
- Can be disabled with `DISABLE_NLP=1` (useful for tests, airgapped envs)
- Graceful fallback to heuristics if transformers not installed or model load fails
- Model: `michellejieli/nsfw_text_classifier` (Hugging Face)

#### Fuzzy Matching Strategy
- **SimHash**: 64-bit locality-sensitive hash using character n-grams (size 3)
  - Hamming distance ≤ 5 indicates match
  - Catches typos, spacing variations
- **Jaccard**: Character n-gram set similarity (≥ 0.80 threshold)
  - Effective for letter substitutions
- **Soundex**: Phonetic encoding for homophones
  - Catches "loli" vs "lolee", "kid" vs "kyd"

#### Rate Limiting
- In-memory token bucket per user ID (client IP or X-Forwarded-For)
- Thread-safe with explicit locking (for FastAPI async context)
- Automatic cleanup of stale entries (prevents memory leak)
- Configurable: max_requests (100) per window (60s)

#### Testing Patterns
- **NLP Disabled by Default**: Tests run with `DISABLE_NLP=1` for speed
- **Fixture Organization**: conftest.py provides shared fixtures
- **Bug Regression Tests**: Separate test files for historical bugs (test_bug_*.py)
- **Coverage Target**: >80% for new code (enforced in CI)

#### RSS Term Updates (`/update_terms` endpoint)
- Fetches child safety RSS feeds
- Extracts candidate terms using NLP or keyword fallback
- Adds to pending_terms with TTL (24h)
- Promotes to active terms after confirmation threshold (2 sources)
- Expires stale pending terms automatically

### Decision Object Structure

```python
Decision(
    allow: bool,           # True to allow, False to block
    action: str,           # "ALLOW" or "BLOCK"
    reason: str,           # Human-readable explanation
    normalized_prompt: str, # Normalized version of input
    rewritten_prompt: Optional[str],  # Fun rewrite (if requested)
    signals: Dict[str, Any]  # All detection signals (ages, terms, scores, severity)
)
```

**Signals Dictionary** contains:
- `normalized`: Normalized text
- `ages`: List of detected ages
- `hard_terms`, `ambiguous_youth`, `adult_assertions`, `injections`: Sets of matched terms
- `school_context`, `cross_sentence`: Additional context matches
- `second_pass`: Fuzzy match results with window, match_type, term, distance/similarity
- `context_score`: Calculated risk score
- `nlp_flagged`: Whether NLP classified as NSFW
- `severity`: Severity level (CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN)
- `allowlisted`: Professional context flag

### Metrics and Observability

**Prometheus Metrics** (when enabled):
- `csam_requests_total{endpoint}` - Total requests by endpoint
- `csam_decisions_total{action, category}` - Decisions by action and severity
- `csam_rss_terms_total{category}` - RSS terms proposed
- `csam_rss_term_churn{action, category}` - Term additions/expirations

**Logging**:
- Uses Python logging module
- Decision objects can be serialized to JSON (`to_json()`)
- No raw content logged (privacy-preserving hashes only)

### Text Normalization Pipeline

Normalization is multi-stage to handle evasion techniques:

1. **Strip and Unicode Normalization**
   - Remove zero-width characters, line/paragraph separators
   - Fold homoglyphs (Cyrillic 'а' → Latin 'a', fullwidth chars → ASCII)
   - Apply NFKC normalization (compatibility decomposition + composition)
   - Remove combining marks

2. **Leet Speak Conversion**
   - Translate common substitutions: 0→o, 1→i, 3→e, 4→a, 5→s, 7→t, @→a, $→s

3. **Repeated Character Reduction**
   - "loooooli" → "looli" (max 2 consecutive chars)

4. **Whitespace Normalization**
   - Collapse multiple spaces to single space
   - Lowercase all text

5. **Obfuscation Detection**
   - Squashed text: remove non-word chars between letters ("l!o!l!i" → "loli")
   - Deobfuscated: remove all spaces ("l o l i" → "loli")

### Severity Levels

Terms are classified by severity for prioritized response:

- **CRITICAL**: loli, shota, child porn, pedo, pedophile, toddler, infant
- **HIGH**: kid, minor, underage, preteen, child, children
- **MEDIUM**: teen, teenage, jailbait, barely legal
- **LOW**: youthful, nubile

Decisions include severity in signals for downstream filtering/alerting.

## Working with This Codebase

### When Adding New Terms
1. Add to appropriate list in `DEFAULT_CONFIG` (hard_terms, ambiguous_youth, etc.)
2. Assign weight if ambiguous term (in ambiguous_youth_weights)
3. Consider adding to severity_levels for classification
4. Update tests to verify detection
5. Rebuild term regexes are cached (LRU cache) so changes need restart

### When Modifying Detection Logic
1. Update `_extract_signals` for new signal types
2. Update `_make_decision` to incorporate new signals
3. Adjust `_context_score` if new weighted factors added
4. Add corresponding test cases in tests/test_guard_extended.py
5. Consider impact on false positive rate (check against test suite)

### When Working with Images
- Perceptual hash computation is in `_compute_phash`
- Hash comparison uses Hamming distance (bit differences)
- Known hashes loaded from JSON file at startup (HASH_LIST_PATH env var)
- Hash format: 16-character hex string (64-bit)
- Pillow safety settings: LOAD_TRUNCATED_IMAGES=True, MAX_IMAGE_PIXELS=64M

### When Running Tests Locally
- Always use `DISABLE_NLP=1` unless specifically testing NLP
- Use `PYTHONPATH=src` to ensure editable install works
- Coverage reports generated in terminal and XML
- CI runs on Python 3.10, 3.11, 3.12 across Ubuntu/macOS/Windows
- Docker build tested in CI (docker/Dockerfile.api)

### When Debugging Decisions
1. Enable verbose mode: `assess(prompt, verbose=True)`
2. Check logger output for normalized/squashed text
3. Inspect `Decision.signals` dictionary for all detection details
4. Use `/version` endpoint to verify model loaded
5. Test isolated components: `_normalize_text`, `_find_terms_regex`, `_second_pass_detect`

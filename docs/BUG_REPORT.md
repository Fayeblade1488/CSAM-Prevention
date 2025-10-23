# Bug Report for CSAM-Prevention Repository

## Date: 2025-10-23

## Methodology
Systematic code review of all source files focusing on:
- Logic errors
- Edge cases
- Type safety issues
- Error handling
- Configuration validation
- Data flow issues
- Security concerns

---

## MAJOR BUGS (5)

### Bug 1: FastAPI Deprecation Warning - Lifespan Event Handler
**File**: `src/csam_guard/app.py`  
**Lines**: 28-43  
**Severity**: MEDIUM  
**Type**: Deprecation

**Description**:
The application uses the deprecated `@app.on_event("startup")` decorator which will be removed in future versions of FastAPI. This is currently generating warnings during test runs.

**Impact**:
- Breaking change in future FastAPI versions
- Technical debt accumulation
- Deprecation warnings clutter test output

**Current Code**:
```python
@app.on_event("startup")
async def startup_event():
    """Initializes the CSAMGuard instance at application startup."""
    # ... code
```

**Proposed Fix**:
Replace with modern lifespan context manager:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    hash_path = os.getenv("HASH_LIST_PATH")
    # ... initialization code
    yield
    # Shutdown (if needed)

app = FastAPI(title="CSAM Guard API", lifespan=lifespan)
```

---

### Bug 2: Potential Integer Overflow in Perceptual Hash Computation
**File**: `src/csam_guard/guard.py`  
**Lines**: 777-810  
**Severity**: LOW  
**Type**: Edge Case

**Description**:
The `_compute_phash` function builds a hash by left-shifting bits. While Python handles arbitrary precision integers, the hash is expected to be 64-bit. The loop constructs the hash starting from 0 with 64 bits from the DCT coefficients.

**Impact**:
- Inconsistent hash values on different Python versions or platforms
- Potential hash collisions
- Comparison failures with stored hashes

**Current Code**:
```python
bits = (dct_low > med).astype(np.uint8)
bits = np.concatenate([[0], bits])
h = 0
for b in bits:
    h = (h << 1) | int(b)
return h
```

**Concern**:
The bits array has 64 elements (63 DCT coefficients + 1 prepended 0), but the loop creates a value that could theoretically exceed 64 bits if not properly constrained.

**Proposed Fix**:
Explicitly mask to 64 bits:
```python
h = 0
for b in bits:
    h = ((h << 1) | int(b)) & 0xFFFFFFFFFFFFFFFF
return h
```

---

### Bug 3: Race Condition in Rate Limiter
**File**: `src/csam_guard/guard.py`  
**Lines**: 180-206  
**Severity**: MEDIUM  
**Type**: Concurrency

**Description**:
The `RateLimiter` class uses a simple in-memory `defaultdict(list)` to track requests. In a multi-threaded or async environment (which FastAPI uses), this can lead to race conditions where multiple requests from the same user are processed simultaneously and may not properly enforce the rate limit.

**Impact**:
- Rate limit bypass under high concurrent load
- Inconsistent rate limiting behavior
- Potential DoS vulnerability

**Current Code**:
```python
def check(self, user_id: str) -> bool:
    now = time()
    self.requests[user_id] = [t for t in self.requests[user_id] if now - t < self.window]
    if len(self.requests[user_id]) >= self.max_requests:
        return False
    self.requests[user_id].append(now)
    return True
```

**Proposed Fix**:
Add thread-safe locking:
```python
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
        self.lock = Lock()
    
    def check(self, user_id: str) -> bool:
        with self.lock:
            now = time()
            self.requests[user_id] = [t for t in self.requests[user_id] if now - t < self.window]
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            self.requests[user_id].append(now)
            return True
```

---

### Bug 4: Unsafe MD5 Usage in SimHash
**File**: `src/csam_guard/guard.py`  
**Lines**: 575-604  
**Severity**: LOW  
**Type**: Security (informational)

**Description**:
The `_simhash` function uses MD5 for hashing n-grams. While MD5 is cryptographically broken for collision resistance, in this context it's used for fingerprinting, not security. However, Python emits security warnings about MD5 usage in FIPS mode.

**Impact**:
- Potential issues in FIPS-compliant environments
- Security scanner false positives
- Code quality concerns

**Current Code**:
```python
h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
```

**Proposed Fix**:
Use a non-cryptographic hash function or explicitly acknowledge the non-security use:
```python
# Use SHA-256 (slower but more acceptable)
h = int(hashlib.sha256(g.encode("utf-8")).hexdigest(), 16)

# OR use Python's built-in hash (faster, non-cryptographic)
h = hash(g) & 0xFFFFFFFFFFFFFFFF  # mask to 64 bits
```

---

### Bug 5: Missing Error Handling in Image Assessment
**File**: `src/csam_guard/guard.py`  
**Lines**: 812-842  
**Severity**: MEDIUM  
**Type**: Error Handling

**Description**:
The `assess_image` function catches `DecompressionBombError` and general `Exception`, but doesn't handle all PIL-specific errors like `UnidentifiedImageError` (when PIL can't identify the image format) or `ValueError` (for invalid image data). These exceptions can leak through and cause unhandled errors.

**Impact**:
- Unhandled exceptions crash the API endpoint
- Poor error messages for users
- Difficulty debugging image processing issues

**Current Code**:
```python
try:
    if image_path:
        img = Image.open(image_path)
    else:
        img = Image.open(BytesIO(image_data))
except Image.DecompressionBombError:
    # ... handle
except Exception as e:
    # ... handle
```

**Proposed Fix**:
Add more specific exception handling:
```python
try:
    if image_path:
        img = Image.open(image_path)
    else:
        img = Image.open(BytesIO(image_data))
except Image.DecompressionBombError:
    decision = Decision(allow=False, action="BLOCK", reason="Image exceeds decompression limits", normalized_prompt=normalized, signals={"error":"DecompressionBombError"})
    self._api_log(decision)
    return decision
except Image.UnidentifiedImageError as e:
    decision = Decision(allow=False, action="BLOCK", reason="Unidentified or corrupt image format", normalized_prompt=normalized, signals={"error":str(e)})
    self._api_log(decision)
    return decision
except ValueError as e:
    decision = Decision(allow=False, action="BLOCK", reason=f"Invalid image data: {str(e)}", normalized_prompt=normalized, signals={"error":str(e)})
    self._api_log(decision)
    return decision
except Exception as e:
    decision = Decision(allow=False, action="BLOCK", reason=f"Image processing error: {str(e)}", normalized_prompt=normalized, signals={"error":str(e)})
    self._api_log(decision)
    return decision
```

---

## MINOR BUGS (10)

### Minor Bug 1: Inconsistent Set vs List Conversion in Decision Signals
**File**: `src/csam_guard/guard.py`  
**Lines**: 700, 706, 715, 719, 731, 958  
**Severity**: LOW  
**Type**: Data Consistency

**Description**:
Multiple places in `_make_decision` convert sets to sorted lists for the signals dict: `{k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()}`. This is done inline multiple times, which is inefficient and error-prone.

**Impact**:
- Code duplication
- Potential inconsistency if one location is updated but not others
- Performance overhead from repeated conversions

**Proposed Fix**:
Extract to a helper method:
```python
def _serialize_signals(self, signals: Dict) -> Dict:
    """Converts sets to sorted lists for JSON serialization."""
    return {k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()}
```

---

### Minor Bug 2: Hardcoded Prompt Length Limit
**File**: `src/csam_guard/guard.py`  
**Lines**: 314, 617  
**Severity**: LOW  
**Type**: Magic Number

**Description**:
The RSS feed content and NLP classifier both use a hardcoded `[:4000]` character limit. This should be a configurable parameter.

**Impact**:
- Inflexibility for different use cases
- Potential truncation of important context
- Inconsistent behavior if limits change

**Proposed Fix**:
Add to DEFAULT_CONFIG:
```python
"max_text_length": 4000,
```

---

### Minor Bug 3: Missing Validation for phash_match_thresh
**File**: `src/csam_guard/guard.py`  
**Lines**: 854  
**Severity**: LOW  
**Type**: Configuration Validation

**Description**:
The `phash_match_thresh` config value is not validated to ensure it's within a reasonable range (0-64 for 64-bit hashes). Invalid values could cause incorrect matching behavior.

**Impact**:
- False positives/negatives in image matching
- Confusing behavior with invalid configuration

**Proposed Fix**:
Add validation in `_validate_config`:
```python
if not (0 <= config["phash_match_thresh"] <= 64):
    raise ValueError(f"phash_match_thresh must be between 0 and 64, got {config['phash_match_thresh']}")
```

---

### Minor Bug 4: Potential KeyError in Metrics Recording
**File**: `src/csam_guard/guard.py`  
**Lines**: 172  
**Severity**: LOW  
**Type**: Error Handling

**Description**:
`self.severity_counts[decision.signals.get("severity", "UNKNOWN")]` could potentially fail if signals is None or not a dict, though this should be prevented by the code structure.

**Impact**:
- Rare edge case crash
- Poor error reporting

**Proposed Fix**:
Add defensive check:
```python
severity = "UNKNOWN"
if isinstance(decision.signals, dict):
    severity = decision.signals.get("severity", "UNKNOWN")
self.severity_counts[severity] += 1
```

---

### Minor Bug 5: No Cleanup for RateLimiter Old Entries
**File**: `src/csam_guard/guard.py`  
**Lines**: 180-206  
**Severity**: LOW  
**Type**: Memory Leak

**Description**:
The RateLimiter never removes user_id keys from the dict once they're added, even if the user hasn't made a request in a long time. This causes unbounded memory growth over time.

**Impact**:
- Memory leak in long-running services
- Performance degradation over time

**Proposed Fix**:
Add periodic cleanup or limit dict size:
```python
def check(self, user_id: str) -> bool:
    now = time()
    # Clean up old users periodically
    if len(self.requests) > 10000:  # arbitrary limit
        self._cleanup_old_entries(now)
    self.requests[user_id] = [t for t in self.requests[user_id] if now - t < self.window]
    # ... rest of method

def _cleanup_old_entries(self, now: float):
    """Remove users who haven't made requests recently."""
    to_remove = [uid for uid, times in self.requests.items() 
                 if not times or now - times[-1] > self.window * 2]
    for uid in to_remove:
        del self.requests[uid]
```

---

### Minor Bug 6: Soundex Returns "Z000" for Empty/Non-Alpha Strings
**File**: `src/csam_guard/guard.py`  
**Lines**: 632-648  
**Severity**: LOW  
**Type**: Edge Case

**Description**:
The Soundex implementation returns "Z000" for empty or non-alphabetic strings. This could lead to false matches where all non-alpha strings have the same Soundex code.

**Impact**:
- False positives in fuzzy matching
- Unexpected matching behavior for special characters

**Proposed Fix**:
Return None or a special marker for invalid inputs:
```python
s = re.sub(r"[^a-zA-Z]", "", s).upper()
if not s:
    return None  # or ""
```

And update `_soundex_match` to handle None.

---

### Minor Bug 7: Missing Type Validation in _words_to_int
**File**: `src/csam_guard/guard.py`  
**Lines**: 431-450  
**Severity**: LOW  
**Type**: Input Validation

**Description**:
`_words_to_int` doesn't validate that the input is a string, which could cause errors if called with invalid data.

**Impact**:
- Potential crashes with malformed input
- Poor error messages

**Proposed Fix**:
Add input validation:
```python
def _words_to_int(self, phrase: str) -> Optional[int]:
    if not isinstance(phrase, str):
        return None
    # ... rest of implementation
```

---

### Minor Bug 8: Ambiguous Behavior with Empty Age Set
**File**: `src/csam_guard/guard.py`  
**Lines**: 638, 781  
**Severity**: LOW  
**Type**: Logic

**Description**:
`any(a >= 18 for a in signals["ages"])` returns False when ages is an empty set, which is technically correct but semantically ambiguous. The code treats "no ages mentioned" the same as "ages mentioned but all under 18".

**Impact**:
- Potential false positives
- Unclear intent in code

**Proposed Fix**:
Explicitly check for empty ages:
```python
has_explicit_adult_age = bool(signals["ages"]) and any(a >= 18 for a in signals["ages"])
has_explicit_minor_age = bool(signals["ages"]) and any(a < 18 for a in signals["ages"])
```

---

### Minor Bug 9: Potential Division by Zero in Metrics
**File**: `src/csam_guard/guard.py`  
**Lines**: 178  
**Severity**: LOW  
**Type**: Math Error

**Description**:
While protected by `max(1, self.total_requests)`, the block_rate calculation could be misleading when total_requests is 0 (it would show 0/1 = 0.0 rate).

**Impact**:
- Misleading metrics display
- Potential confusion in monitoring

**Proposed Fix**:
Return None or a special value for undefined metrics:
```python
def summary(self) -> Dict:
    block_rate = self.blocks / self.total_requests if self.total_requests > 0 else None
    return {
        "total": self.total_requests,
        "blocks": self.blocks,
        "allows": self.allows,
        "block_rate": block_rate,
        # ...
    }
```

---

### Minor Bug 10: Inconsistent Logging Levels
**File**: `src/csam_guard/guard.py`  
**Lines**: 252, 262, 269, 329, 338, 360, 622  
**Severity**: LOW  
**Type**: Logging Consistency

**Description**:
The code uses a mix of `logger.info`, `logger.warning`, and `logger.error` inconsistently. For example, NLP model load success uses `info`, but failure uses `error` even though the system gracefully falls back to heuristics.

**Impact**:
- Confusing log output
- Difficulty in log analysis and monitoring
- Alert fatigue from "error" logs that aren't really errors

**Proposed Fix**:
Standardize logging levels:
- INFO: Normal operations, successful loads
- WARNING: Degraded operation (NLP disabled), configuration issues
- ERROR: Actual errors that prevent operation

```python
except Exception as e:
    self.classifier = None
    self.logger.warning(f"Failed to load NLP model {model_name}: {e}. Using heuristics only.")
```

---

## SUMMARY

Total Bugs Identified: 15 (5 major + 10 minor)

### Priority for Fixing:
1. **Bug 1 (FastAPI Deprecation)** - High priority, affects maintainability
2. **Bug 3 (Race Condition)** - High priority, affects production reliability
3. **Bug 5 (Image Error Handling)** - Medium priority, affects user experience
4. **Bug 2 (Perceptual Hash)** - Medium priority, affects correctness
5. **Minor bugs** - Low priority, can be addressed incrementally

### Recommended Immediate Actions:
1. Fix FastAPI deprecation (Bug 1)
2. Add thread-safety to RateLimiter (Bug 3)
3. Improve image error handling (Bug 5)
4. Add regression tests for all fixes


"""Tests for bug fixes implemented in the repository."""
import concurrent.futures
import pytest
from csam_guard.guard import CSAMGuard, DEFAULT_CONFIG, RateLimiter
import time


def test_rate_limiter_thread_safety():
    """Test that RateLimiter is thread-safe under concurrent load.
    
    Regression test for Bug 3: Race Condition in Rate Limiter.
    Before fix: Multiple threads could bypass the rate limit.
    After fix: Thread-safe locking prevents race conditions.
    """
    limiter = RateLimiter(max_requests=5, window=1)
    user_id = "test_user"
    
    def make_request():
        """Simulate a request from the same user."""
        return limiter.check(user_id)
    
    # Try to make 20 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(20)]
        results = [f.result() for f in futures]
    
    # Only 5 should be allowed (max_requests=5)
    allowed = sum(results)
    assert allowed == 5, f"Expected exactly 5 allowed requests, got {allowed}"
    
    # After waiting for the window to expire, should allow 5 more
    time.sleep(1.1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in futures]
    
    allowed = sum(results)
    assert allowed == 5, f"Expected 5 more allowed requests after window expiry, got {allowed}"


def test_rate_limiter_memory_cleanup():
    """Test that RateLimiter cleans up old entries to prevent memory leak.
    
    Regression test for Minor Bug 5: No Cleanup for RateLimiter Old Entries.
    Before fix: Dict would grow unbounded with user IDs.
    After fix: Periodic cleanup removes inactive users.
    """
    limiter = RateLimiter(max_requests=1, window=1)
    
    # Create exactly 10000 user entries
    for i in range(10000):
        limiter.check(f"user_{i}")
    
    # Wait for entries to age out
    time.sleep(2.5)  # window * 2 + 0.5
    
    # Create 51 more entries to trigger cleanup (total > 10000)
    for i in range(10000, 10051):
        limiter.check(f"user_{i}")
    
    # The limiter should have cleaned up old entries during one of the checks
    # Since we made 51 requests after waiting, cleanup should have triggered
    assert len(limiter.requests) < 10051, f"Memory cleanup should have removed some old entries, got {len(limiter.requests)}"
    
    # Most old entries should be gone (only recent ones should remain)
    # We expect around 51 entries (the ones we just added)
    assert len(limiter.requests) < 200, f"Expected much fewer entries after cleanup, got {len(limiter.requests)}"


def test_rate_limiter_per_user_isolation():
    """Test that rate limiting is properly isolated per user.
    
    Ensures that one user hitting the limit doesn't affect other users.
    """
    limiter = RateLimiter(max_requests=3, window=1)
    
    # User A makes 3 requests (hits limit)
    user_a_results = [limiter.check("user_a") for _ in range(4)]
    assert sum(user_a_results) == 3, "User A should be limited to 3 requests"
    
    # User B should still be able to make requests
    user_b_results = [limiter.check("user_b") for _ in range(3)]
    assert sum(user_b_results) == 3, "User B should not be affected by User A's limit"
    
    # User A tries again (should be blocked)
    assert not limiter.check("user_a"), "User A should still be blocked"
    
    # Wait for window to expire
    time.sleep(1.1)
    
    # Both users should be able to make requests again
    assert limiter.check("user_a"), "User A should be unblocked after window"
    assert limiter.check("user_b"), "User B should still be able to make requests"


def test_fastapi_lifespan_no_deprecation():
    """Test that the FastAPI app uses lifespan instead of deprecated on_event.
    
    Regression test for Bug 1: FastAPI Deprecation Warning.
    This test ensures no deprecation warnings are emitted.
    """
    # Import the app to check for deprecation warnings
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from csam_guard.app import app
        
        # Check that no DeprecationWarning was raised for on_event
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        on_event_warnings = [warning for warning in deprecation_warnings if "on_event" in str(warning.message)]
        
        # We expect no on_event deprecation warnings
        assert len(on_event_warnings) == 0, f"Found {len(on_event_warnings)} on_event deprecation warnings"
    
    # Verify that lifespan is configured
    assert hasattr(app, "router"), "App should have router"
    assert app.router.lifespan_context is not None, "App should have lifespan configured"


def test_guard_initialization_with_invalid_config():
    """Test that CSAMGuard validates configuration properly.
    
    Regression test for configuration validation issues.
    """
    # Missing required keys should raise ValueError
    invalid_config = {}
    
    with pytest.raises(ValueError, match="Config missing keys"):
        CSAMGuard(invalid_config)
    
    # Valid config should work
    valid_config = DEFAULT_CONFIG.copy()
    guard = CSAMGuard(valid_config)
    assert guard is not None


def test_rate_limiter_edge_cases():
    """Test edge cases in rate limiter behavior."""
    limiter = RateLimiter(max_requests=2, window=1)
    
    # Test with empty user_id (edge case)
    assert limiter.check(""), "Empty user_id should be allowed"
    
    # Test with very long user_id
    long_id = "x" * 1000
    assert limiter.check(long_id), "Long user_id should be allowed"
    
    # Test rapid succession
    user = "rapid_user"
    results = []
    for _ in range(5):
        results.append(limiter.check(user))
    
    # Should allow exactly max_requests
    assert sum(results) == 2, f"Expected 2 allowed in rapid succession, got {sum(results)}"


def test_rate_limiter_window_boundaries():
    """Test rate limiter behavior at window boundaries."""
    limiter = RateLimiter(max_requests=2, window=1)
    user = "boundary_user"
    
    # Make 2 requests at t=0
    assert limiter.check(user), "First request should be allowed"
    assert limiter.check(user), "Second request should be allowed"
    assert not limiter.check(user), "Third request should be blocked"
    
    # Wait just under the window
    time.sleep(0.9)
    assert not limiter.check(user), "Request before window expiry should be blocked"
    
    # Wait for full window to expire
    time.sleep(0.2)  # Total 1.1 seconds
    assert limiter.check(user), "Request after window expiry should be allowed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

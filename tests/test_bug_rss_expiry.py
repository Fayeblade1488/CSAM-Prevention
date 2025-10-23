"""Test to demonstrate the RSS pending terms expiry bug."""
import pytest
from csam_guard.guard import CSAMGuard, DEFAULT_CONFIG
from datetime import datetime, timedelta


def test_rss_pending_terms_silent_deletion_bug():
    """Test that demonstrates a bug in RSS pending terms expiry logic.
    
    Bug: In the update_terms_from_rss method, when checking for expired terms,
    any exception (e.g., missing 'added' key, malformed timestamp) causes the
    term to be silently deleted via:
    
        except Exception:
            expired.append(term)
    
    This means legitimate pending terms with corrupted metadata are incorrectly
    removed instead of being preserved or logged for investigation.
    """
    config = DEFAULT_CONFIG.copy()
    
    # Add a pending term with malformed data (missing 'added' key)
    config["pending_terms"]["hard_terms"]["testterm"] = {
        "sources": ["http://example.com"],
        "weight": 2
        # Missing 'added' key!
    }
    
    guard = CSAMGuard(config)
    
    # Verify the term exists
    assert "testterm" in guard.config["pending_terms"]["hard_terms"]
    
    # Call update_terms_from_rss which will check for expiry
    # This should not make any HTTP requests since we're using default config
    # It will just check expiry of existing pending terms
    guard.update_terms_from_rss()
    
    # After the fix, the term should NOT be deleted - it should be preserved
    is_deleted = "testterm" not in guard.config["pending_terms"]["hard_terms"]
    print(f"Term was deleted due to malformed data: {is_deleted}")
    
    # With the fix, the term is preserved
    assert not is_deleted, "Bug fixed: term with malformed timestamp is preserved"


def test_rss_pending_terms_valid_expiry():
    """Test that valid terms with proper timestamps are expired correctly."""
    config = DEFAULT_CONFIG.copy()
    
    # Add a pending term that should be expired (added 25 hours ago, TTL is 24 hours)
    old_time = (datetime.now() - timedelta(hours=25)).isoformat()
    config["pending_terms"]["hard_terms"]["oldterm"] = {
        "added": old_time,
        "sources": ["http://example.com"],
        "weight": 2
    }
    
    # Add a pending term that should NOT be expired (added 1 hour ago)
    recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
    config["pending_terms"]["hard_terms"]["newterm"] = {
        "added": recent_time,
        "sources": ["http://example.com"],
        "weight": 2
    }
    
    guard = CSAMGuard(config)
    
    # Verify both terms exist
    assert "oldterm" in guard.config["pending_terms"]["hard_terms"]
    assert "newterm" in guard.config["pending_terms"]["hard_terms"]
    
    # Update terms which checks for expiry
    guard.update_terms_from_rss()
    
    # Old term should be expired and removed
    assert "oldterm" not in guard.config["pending_terms"]["hard_terms"]
    
    # New term should still exist
    assert "newterm" in guard.config["pending_terms"]["hard_terms"]


def test_rss_pending_terms_invalid_timestamp_format():
    """Test handling of invalid timestamp format."""
    config = DEFAULT_CONFIG.copy()
    
    # Add a term with invalid timestamp format
    config["pending_terms"]["hard_terms"]["badtimestamp"] = {
        "added": "not-a-valid-timestamp",
        "sources": ["http://example.com"],
        "weight": 2
    }
    
    guard = CSAMGuard(config)
    assert "badtimestamp" in guard.config["pending_terms"]["hard_terms"]
    
    # Update terms
    guard.update_terms_from_rss()
    
    # After the fix, this term should NOT be deleted
    is_deleted = "badtimestamp" not in guard.config["pending_terms"]["hard_terms"]
    print(f"Term with invalid timestamp was deleted: {is_deleted}")
    
    # With the fix, the term is preserved
    assert not is_deleted, "Bug fixed: term with invalid timestamp is preserved"

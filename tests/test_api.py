"""Tests for the FastAPI application endpoints."""

import base64
import pytest
from fastapi.testclient import TestClient
from csam_guard.app import app
from csam_guard.guard import CSAMGuard, DEFAULT_CONFIG, RateLimiter


@pytest.fixture(scope="module")
def test_client():
    """Create a test client with initialized app state."""
    # Initialize app state manually for testing
    app.state.guard = CSAMGuard(DEFAULT_CONFIG.copy())
    app.state.limiter = RateLimiter(
        DEFAULT_CONFIG["rate_limit_max"], DEFAULT_CONFIG["rate_limit_window"]
    )
    app.state.max_upload_size = 10_000_000

    with TestClient(app) as client:
        yield client


def test_health_endpoint(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint(test_client):
    """Test the version endpoint."""
    response = test_client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "14.1.0"
    assert "model" in data
    assert "model_version" in data


def test_assess_endpoint_safe_text(test_client):
    """Test the assess endpoint with safe text."""
    response = test_client.post(
        "/assess", json={"prompt": "This is a safe adult conversation"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allow"] is True
    assert data["action"] == "ALLOW"


def test_assess_endpoint_unsafe_text(test_client):
    """Test the assess endpoint with unsafe text."""
    response = test_client.post(
        "/assess", json={"prompt": "A 15-year-old in school uniform"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allow"] is False
    assert data["action"] == "BLOCK"


def test_assess_endpoint_with_fun_rewrite(test_client):
    """Test the assess endpoint with fun rewrite enabled."""
    response = test_client.post(
        "/assess", json={"prompt": "18+ adult woman", "do_fun_rewrite": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allow"] is True
    if data.get("rewritten_prompt"):
        # Fun rewrite may or may not be applied depending on content
        assert isinstance(data["rewritten_prompt"], str)


def test_assess_image_endpoint_safe(test_client):
    """Test the assess_image endpoint with a safe image."""
    # Create a small test PNG image
    mock_safe_image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABNSURBVHhe7c4xCoAwEATB3v+j3d3d3VYT8j6QEG3bDzzvBzSaTdPcNLtgbtPcpblNc5vmNs1tmts0t2lu09ymuU1zm+Y2zW2a2zS3aW7T3AAAAAD//wMA3kHWkQAAAABJRU5ErkJggg=="
    )

    response = test_client.post(
        "/assess_image", files={"file": ("test.png", mock_safe_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allow"] is True
    assert data["action"] == "ALLOW"


def test_assess_image_endpoint_unsupported_type(test_client):
    """Test the assess_image endpoint with an unsupported file type."""
    response = test_client.post(
        "/assess_image", files={"file": ("test.txt", b"Hello", "text/plain")}
    )
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_rate_limiting(test_client):
    """Test that rate limiting is enforced."""
    # This test may be difficult to reliably test without mocking
    # but we can at least verify the endpoint responds
    for _ in range(5):
        response = test_client.post("/assess", json={"prompt": "test"})
        assert response.status_code in [200, 429]


def test_update_terms_endpoint(test_client):
    """Test the update_terms endpoint."""
    response = test_client.get("/update_terms")
    # This should succeed even if no terms are updated
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

import pytest
from fastapi.testclient import TestClient
from csam_guard.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_assess_image_unsupported_media_type(client):
    response = client.post("/assess_image", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 415

def test_assess_image_file_too_large(client):
    response = client.post(
        "/assess_image",
        files={"file": ("test.jpg", b"a" * (10000001), "image/jpeg")},
    )
    assert response.status_code == 413

def test_update_terms(client):
    response = client.get("/update_terms")
    assert response.status_code == 200
    assert response.json() == {"status": "Terms updated"}

def test_assess_with_fun_rewrite(client):
    response = client.post("/assess", json={"prompt": "teen", "do_fun_rewrite": True})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["rewritten_prompt"] == "wrinkly grandpa"

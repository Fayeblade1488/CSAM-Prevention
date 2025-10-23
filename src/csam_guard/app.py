"""This module contains the FastAPI application for the CSAM Guard service.

It defines the API endpoints for assessing text and images, as well as for
health checks, version information, and updating terms from RSS feeds. It also
handles the application startup logic, including the initialization of the
CSAMGuard instance.
"""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Request, status
from pydantic import BaseModel
from .guard import CSAMGuard, DEFAULT_CONFIG, RateLimiter, ALLOWED_IMAGE_CT
from prometheus_client import make_asgi_app

PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "0") == "1"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10000000"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))

app = FastAPI(title="CSAM Guard API")
app.state.limiter = RateLimiter(
    DEFAULT_CONFIG["rate_limit_max"], DEFAULT_CONFIG["rate_limit_window"]
)
app.state.max_upload_size = MAX_UPLOAD_BYTES

if PROMETHEUS_ENABLED:
    app.mount("/metrics", make_asgi_app())

@app.on_event("startup")
async def startup_event():
    """Initializes the CSAMGuard instance at application startup."""
    # Allow reading known hashes from file path if provided
    hash_path = os.getenv("HASH_LIST_PATH")
    conf = DEFAULT_CONFIG.copy()
    if hash_path and os.path.exists(hash_path):
        try:
            import json
            with open(hash_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "known_csam_phashes" in data:
                conf["known_csam_phashes"] = data["known_csam_phashes"]
        except Exception:
            pass
    app.state.guard = CSAMGuard(conf)

def check_rate_limit(request: Request):
    """Checks if the client has exceeded the rate limit.

    Args:
        request: The incoming request.

    Raises:
        HTTPException: If the rate limit is exceeded.
    """
    trust_proxy = os.getenv("TRUST_XFF", "0") == "1"
    user_id = request.client.host
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            user_id = fwd.split(",")[0].strip()
    if not app.state.limiter.check(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.get("/health")
def health():
    """Returns the health status of the service."""
    return {"status": "ok"}

@app.get("/version")
def version():
    """Returns the version of the service and the NLP model."""
    return {"version": "14.1.0", "model": DEFAULT_CONFIG["nlp_model_name"], "model_version": DEFAULT_CONFIG["nlp_model_version"]}

class PromptRequest(BaseModel):
    """The request model for the /assess endpoint."""
    prompt: str
    do_fun_rewrite: bool = False
    verbose: bool = False

@app.post("/assess", dependencies=[Depends(check_rate_limit)])
def assess_prompt(req: PromptRequest):
    """Assesses a text prompt for potential CSAM-related content."""
    return app.state.guard.assess(req.prompt, req.do_fun_rewrite, verbose=req.verbose)

@app.post("/assess_image", dependencies=[Depends(check_rate_limit)])
async def assess_image_endpoint(request: Request, file: UploadFile = File(...)):
    """Assesses an image for potential CSAM-related content."""
    if file.content_type not in ALLOWED_IMAGE_CT:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    cl = request.headers.get("content-length")
    if cl is not None and int(cl) > app.state.max_upload_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    cap = app.state.max_upload_size + 1
    content = await file.read(cap)
    if len(content) > app.state.max_upload_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    return app.state.guard.assess_image(image_data=content)

@app.get("/update_terms", dependencies=[Depends(check_rate_limit)])
def update_terms():
    """Updates the term lists from the configured RSS feeds."""
    guard = app.state.guard
    guard.update_terms_from_rss()
    return {"status": "Terms updated"}

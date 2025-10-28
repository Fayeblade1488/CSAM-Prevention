# API Reference

The CSAM-Prevention service provides a simple REST API for assessing text and images.

## Endpoints

### `POST /assess`
Assesses a text prompt for potential CSAM-related content.

**Request Body:**
```json
{
  "prompt": "The text to assess.",
  "do_fun_rewrite": false,
  "verbose": false
}
```

**Response:**
```json
{
  "allow": true,
  "action": "ALLOW",
  "reason": "No minor risk detected",
  "normalized_prompt": "the text to assess.",
  "rewritten_prompt": null,
  "signals": {}
}
```

### `POST /assess_image`
Assesses an image for potential CSAM-related content.

**Request Body:**
A multipart form data request with a single file field named `file`.

**Response:**
```json
{
  "allow": true,
  "action": "ALLOW",
  "reason": "No CSAM image match detected",
  "normalized_prompt": "image.jpg",
  "signals": {}
}
```

### `GET /health`
Returns the health status of the service.

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /version`
Returns the version of the service and the NLP model.

**Response:**
```json
{
  "version": "14.1.0",
  "model": "michellejieli/nsfw_text_classifier",
  "model_version": "1.0"
}
```

### `GET /update_terms`
Updates the term lists from the configured RSS feeds.

**Response:**
```json
{
  "status": "Terms updated"
}
```

# Chaos Auto Prompt API - Quick Start Guide

## Overview

The FastAPI application has been successfully set up in `/api/main.py` with production-ready configuration.

## Files Created

1. **`api/main.py`** - Main FastAPI application with:
   - Proper title, description, and version
   - CORS middleware using settings configuration
   - Exception handlers for ProviderError, DatasetError, OptimizationError
   - Structured logging (JSON or text format)
   - Startup/shutdown lifespan events
   - Root and health check endpoints
   - Uvicorn run configuration

2. **`api/__init__.py`** - Package initialization exporting the app

3. **`scripts/run_api.py`** - Convenient run script for the API server

## Features Implemented

### 1. CORS Middleware
- Configured via environment variables (settings.cors_*)
- Default origins: http://localhost:3000, http://localhost:4357
- Supports credentials, all methods, and all headers

### 2. Exception Handlers
- **ProviderError** (503): AI provider communication failures
- **DatasetError** (400): Dataset validation/processing errors
- **OptimizationError** (422): Optimization process failures
- **General Exception** (500): Catch-all for unexpected errors

### 3. Logging
- Configurable log level via LOG_LEVEL environment variable
- JSON format for production, text format for development
- Structured logging with timestamps and levels

### 4. OpenAPI Documentation
- Interactive docs at `/docs` (Swagger UI)
- Alternative docs at `/redoc` (ReDoc)
- OpenAPI schema at `/openapi.json`

### 5. Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check with configuration details

## Running the API

### Method 1: Using the run script (Recommended)
```bash
cd /path/to/chaos-auto-prompt
python scripts/run_api.py
```

### Method 2: Using uvicorn directly
```bash
cd /path/to/chaos-auto-prompt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Using python -m
```bash
cd /path/to/chaos-auto-prompt
python -m api.main
```

## Configuration

All configuration is done via environment variables in `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Server (optional, defaults shown)
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'text'

# CORS (optional, defaults shown)
CORS_ORIGINS=["http://localhost:3000","http://localhost:4357"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["*"]
CORS_ALLOW_HEADERS=["*"]
```

## Accessing the API

Once running, access:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Testing

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "default_model": "gpt-4o",
  "log_level": "INFO",
  "cors_origins": ["http://localhost:3000", "http://localhost:4357"]
}
```

## Next Steps

To add new routes:
1. Create route modules in `api/routes/`
2. Import and include routers in `api/main.py`:
   ```python
   from api.routes import optimization
   app.include_router(optimization.router, prefix="/api/v1", tags=["optimization"])
   ```

## Production Deployment

For production deployment:
1. Set `RELOAD=false` in environment
2. Use multiple workers: `WORKERS=4`
3. Use `LOG_FORMAT=json` for structured logging
4. Set appropriate `CORS_ORIGINS` for your domain
5. Use a process manager like systemd or supervisor

Example production command:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## Dependencies

The API requires these packages (already in pyproject.toml):
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- pydantic>=2.5.0
- pydantic-settings>=2.1.0

All other dependencies are from the chaos-auto-prompt package.

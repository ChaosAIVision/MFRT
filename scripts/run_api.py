#!/usr/bin/env python3
"""
Run script for the Chaos Auto Prompt FastAPI application.

This script starts the uvicorn server with the configured settings from
environment variables or defaults.

Usage:
    python scripts/run_api.py

Or with uvicorn directly:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

if __name__ == "__main__":
    # Import after path is set
    from chaos_auto_prompt.config import get_settings

    settings = get_settings()

    print(f"Starting Chaos Auto Prompt API")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Reload: {settings.reload}")
    print(f"  Log level: {settings.log_level}")
    print(f"  Docs: http://{settings.host}:{settings.port}/docs")
    print()

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )

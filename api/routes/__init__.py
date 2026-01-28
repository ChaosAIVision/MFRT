"""
API routes for chaos-auto-prompt.

This package contains all route modules for the FastAPI application.
"""

from .health import router as health_router
from .optimize import router as optimize_router
from .huggingface import router as huggingface_router

__all__ = [
    "health_router",
    "optimize_router",
    "huggingface_router",
]

"""
FastAPI application for chaos-auto-prompt.

This module provides the REST API for prompt optimization using meta-prompting
and evaluation-based optimization strategies.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chaos_auto_prompt.config import get_settings
from chaos_auto_prompt.optimizers.prompt_optimizer import (
    DatasetError,
    OptimizationError,
    ProviderError,
)

# Configure logging
settings = get_settings()
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

# Configure structured logging based on format preference
if settings.log_format == "json":
    # JSON format for production (e.g., with cloud logging)
    logging.basicConfig(
        level=log_level,
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
else:
    # Text format for development
    logging.basicConfig(
        level=log_level,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """
    Manage application lifespan events.

    This function handles startup and shutdown events for the FastAPI application.
    Add any initialization logic here (e.g., database connections, background tasks).

    Args:
        app: The FastAPI application instance

    Yields:
        None
    """
    # Startup: Initialize resources
    logger.info("Starting chaos-auto-prompt API")
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"CORS origins: {settings.cors_origins}")

    # Add any startup initialization here
    # For example: database connections, cache initialization, etc.

    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down chaos-auto-prompt API")

    # Add any cleanup logic here
    # For example: closing database connections, clearing cache, etc.


# Create FastAPI application
app = FastAPI(
    title="Chaos Auto Prompt API",
    description="""
    Production-grade Prompt Learning Optimization SDK with FastAPI REST API.

    This API provides tools for optimizing LLM prompts using various strategies,
    including meta-prompting, evaluation-based optimization, and budget management.

    ## Features

    * **Meta-Prompt Optimization**: Optimize prompts using natural language feedback
    * **Multi-Provider Support**: OpenAI, Google AI, and more
    * **Budget Management**: Track and limit API costs
    * **Batch Processing**: Handle large datasets efficiently
    * **Evaluation Metrics**: Assess prompt performance with various metrics

    ## Usage

    1. Create an optimization request with your prompt and dataset
    2. The API will process your data using the configured model
    3. Receive an optimized prompt with improved performance
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# CORS Middleware
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    """
    Handle provider-related errors.

    ProviderError is raised when there are issues with AI provider interactions
    (e.g., API errors, rate limits, authentication failures).

    Args:
        request: The incoming request
        exc: The ProviderError exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Provider error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "provider_error",
            "message": str(exc),
            "detail": "Error communicating with AI provider. Please try again later.",
        },
    )


@app.exception_handler(DatasetError)
async def dataset_error_handler(request: Request, exc: DatasetError) -> JSONResponse:
    """
    Handle dataset-related errors.

    DatasetError is raised when there are issues with dataset processing
    (e.g., invalid format, missing columns, loading errors).

    Args:
        request: The incoming request
        exc: The DatasetError exception

    Returns:
        JSONResponse with error details
    """
    logger.warning(f"Dataset error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "dataset_error",
            "message": str(exc),
            "detail": "Invalid dataset format or content. Please check your input.",
        },
    )


@app.exception_handler(OptimizationError)
async def optimization_error_handler(
    request: Request, exc: OptimizationError
) -> JSONResponse:
    """
    Handle optimization-related errors.

    OptimizationError is raised when there are issues during the optimization process
    (e.g., convergence failures, budget exceeded, iteration limits).

    Args:
        request: The incoming request
        exc: The OptimizationError exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Optimization error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "optimization_error",
            "message": str(exc),
            "detail": "Failed to optimize prompt. Check your inputs and constraints.",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other unhandled exceptions.

    This is a catch-all handler for any unexpected errors.

    Args:
        request: The incoming request
        exc: The unhandled exception

    Returns:
        JSONResponse with error details
    """
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.log_level == "DEBUG" else "Please contact support",
        },
    )


# =============================================================================
# Routes
# =============================================================================

# Include route modules
from api.routes import health_router, optimize_router, huggingface_router

app.include_router(health_router, tags=["health"])
app.include_router(optimize_router)
app.include_router(huggingface_router)


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """
    Root endpoint.

    Returns basic information about the API and provides links to documentation.

    Returns:
        Dictionary with API information
    """
    return {
        "name": "Chaos Auto Prompt API",
        "version": "0.1.0",
        "description": "Production-grade Prompt Learning Optimization SDK",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns the current status of the API and its configuration.

    Returns:
        Dictionary with health status information
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "default_model": settings.default_model,
        "log_level": settings.log_level,
        "cors_origins": settings.cors_origins,
    }


# =============================================================================
# Run with uvicorn
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )

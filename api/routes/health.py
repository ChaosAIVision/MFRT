"""
Health and status endpoints for the chaos-auto-prompt API.

This module provides endpoints for monitoring the health and status of the API,
including configuration status, provider availability, and model information.
"""

import logging
import os
from time import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from chaos_auto_prompt.config import get_settings
from chaos_auto_prompt.providers import OpenAIProvider, GoogleProvider

from api.models import HealthResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Track server start time
_start_time = time()


def check_api_key_configured(api_key: str) -> bool:
    """
    Check if an API key is properly configured.

    Args:
        api_key: The API key to check

    Returns:
        True if the API key is configured (not empty/min length), False otherwise
    """
    if not api_key or api_key.strip() == "":
        return False
    # Basic check for API key format (most keys are at least 20 characters)
    return len(api_key) >= 20


def get_providers_status(settings) -> Dict[str, str]:
    """
    Get the status of all configured providers.

    Args:
        settings: Application settings

    Returns:
        Dictionary mapping provider names to their status
    """
    providers_status = {}

    # Check OpenAI
    openai_configured = check_api_key_configured(settings.openai_api_key)
    providers_status["openai"] = "configured" if openai_configured else "not_configured"

    # Check Google
    google_configured = check_api_key_configured(settings.google_api_key)
    providers_status["google"] = "configured" if google_configured else "not_configured"

    return providers_status


def get_available_models(settings) -> List[Dict[str, str]]:
    """
    Get list of available models based on configured providers.

    Args:
        settings: Application settings

    Returns:
        List of available model information
    """
    models = []

    # OpenAI models
    if check_api_key_configured(settings.openai_api_key):
        models.extend([
            {"name": "gpt-4o", "provider": "openai"},
            {"name": "gpt-4o-mini", "provider": "openai"},
            {"name": "gpt-4-turbo", "provider": "openai"},
            {"name": "gpt-3.5-turbo", "provider": "openai"},
        ])

    # Google models
    if check_api_key_configured(settings.google_api_key):
        models.extend([
            {"name": "gemini-2.5-flash", "provider": "google"},
            {"name": "gemini-2.5-pro", "provider": "google"},
            {"name": "gemini-1.5-flash", "provider": "google"},
            {"name": "gemini-1.5-pro", "provider": "google"},
        ])

    return models


def get_available_providers(settings) -> List[str]:
    """
    Get list of available (configured) providers.

    Args:
        settings: Application settings

    Returns:
        List of provider names that are configured
    """
    providers = []

    if check_api_key_configured(settings.openai_api_key):
        providers.append("openai")

    if check_api_key_configured(settings.google_api_key):
        providers.append("google")

    return providers


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns the current health status of the API including:
    - Overall status (healthy, degraded, unhealthy)
    - API version
    - Environment name
    - Server uptime
    - Status of external services (providers)

    Returns:
        HealthResponse: Health status information
    """
    settings = get_settings()

    # Determine overall health status
    providers_status = get_providers_status(settings)
    configured_providers = [
        provider for provider, status in providers_status.items()
        if status == "configured"
    ]

    if len(configured_providers) == 0:
        overall_status = "unhealthy"
    elif len(configured_providers) < len(providers_status):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Calculate uptime
    uptime = time() - _start_time

    # Get environment from environment variable
    environment = os.getenv("ENVIRONMENT", "development")

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        environment=environment,
        uptime=uptime,
        services=providers_status,
    )


@router.get("/api/v1/status", tags=["health"])
async def status_check():
    """
    Detailed status endpoint with comprehensive information.

    Returns detailed status information including:
    - Service status (healthy/degraded/unhealthy)
    - Version information
    - Configuration status (API keys loaded)
    - Available providers
    - Available models
    - Default model
    - Server uptime

    Returns:
        Dictionary with detailed status information
    """
    try:
        settings = get_settings()

        # Get providers status
        providers_status = get_providers_status(settings)

        # Determine overall status
        configured_providers = [
            provider for provider, status in providers_status.items()
            if status == "configured"
        ]

        if len(configured_providers) == 0:
            overall_status = "unhealthy"
        elif len(configured_providers) < len(providers_status):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Get available providers and models
        available_providers = get_available_providers(settings)
        available_models = get_available_models(settings)

        # Calculate uptime
        uptime = time() - _start_time

        # Get environment
        environment = os.getenv("ENVIRONMENT", "development")

        # Check API keys configuration
        api_keys_status = {
            "openai": check_api_key_configured(settings.openai_api_key),
            "google": check_api_key_configured(settings.google_api_key),
        }

        return {
            "status": overall_status,
            "version": "0.1.0",
            "environment": environment,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 60)} minutes, {int(uptime % 60)} seconds",
            "configuration": {
                "api_keys_loaded": api_keys_status,
                "default_model": settings.default_model,
                "default_temperature": settings.default_temperature,
                "default_max_tokens": settings.default_max_tokens,
                "log_level": settings.log_level,
                "cors_origins": settings.cors_origins,
            },
            "providers": {
                "available": available_providers,
                "total_count": len(available_providers),
                "status": providers_status,
            },
            "models": {
                "available": available_models,
                "total_count": len(available_models),
                "default": settings.default_model,
            },
        }

    except Exception as e:
        logger.error(f"Error in status check: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving status information: {str(e)}"
        )

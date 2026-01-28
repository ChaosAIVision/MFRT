"""
Model pricing configuration.

Pricing can be overridden via environment variables.
Format: {MODEL_NAME}_INPUT_PRICE and {MODEL_NAME}_OUTPUT_PRICE
Prices are per 1M tokens.
"""

from dataclasses import dataclass
from typing import Dict

from pydantic import Field

from .settings import get_settings


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a specific model."""

    input_price: float  # Price per 1M input tokens
    output_price: float  # Price per 1M output tokens
    model_name: str

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.input_price
        output_cost = (output_tokens / 1_000_000) * self.output_price
        return input_cost + output_cost


# Default pricing configuration (can be overridden by env vars)
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4": ModelPricing(30.0, 60.0, "gpt-4"),
    "gpt-4-turbo": ModelPricing(10.0, 30.0, "gpt-4-turbo"),
    "gpt-4o": ModelPricing(2.50, 10.0, "gpt-4o"),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, "gpt-4o-mini"),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, "gpt-3.5-turbo"),
    "o1": ModelPricing(15.0, 60.0, "o1"),
    "o3": ModelPricing(15.0, 60.0, "o3"),
    "gpt-4.1-nano-2025-04-14": ModelPricing(0.10, 0.40, "gpt-4.1-nano-2025-04-14"),  # Custom model
    "gemini-2.5-flash": ModelPricing(0.075, 0.30, "gemini-2.5-flash"),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, "gemini-2.5-pro"),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, "gemini-1.5-flash"),
    "gemini-1.5-pro": ModelPricing(0.35, 1.05, "gemini-1.5-pro"),
}


def get_model_pricing(model_name: str) -> ModelPricing:
    """
    Get pricing for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        ModelPricing: Pricing information for the model

    Raises:
        ValueError: If model pricing is not found
    """
    settings = get_settings()

    # Check if pricing is overridden by environment variables
    env_input_price = getattr(settings, f"{model_name}_input_price", None)
    env_output_price = getattr(settings, f"{model_name}_output_price", None)

    if env_input_price is not None and env_output_price is not None:
        return ModelPricing(env_input_price, env_output_price, model_name)

    # Use default pricing
    if model_name in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[model_name]

    raise ValueError(f"Pricing not found for model: {model_name}")


def list_supported_models() -> list[str]:
    """Get list of supported models."""
    return list(DEFAULT_MODEL_PRICING.keys())


# Token limit configuration
MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o3": 200000,
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-pro": 2000000,
}


def get_model_context_size(model_name: str) -> int:
    """
    Get context window size for a model.

    Args:
        model_name: Name of the model

    Returns:
        int: Context window size in tokens
    """
    return MODEL_CONTEXT_SIZES.get(model_name, 128000)

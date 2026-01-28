"""
Token pricing calculator for different AI providers with budget tracking.
"""

from typing import Dict, Optional

from ..config.pricing import get_model_pricing
from ..config.settings import get_settings


class PricingCalculator:
    """
    Calculate costs for different AI models with budget tracking.

    This class provides functionality to:
    - Calculate costs for AI model usage
    - Track total usage across multiple calls
    - Monitor budget limits
    - Provide usage summaries
    """

    def __init__(self, budget_limit: Optional[float] = None):
        """
        Initialize the pricing calculator.

        Args:
            budget_limit: Optional budget limit. If not provided, uses default_budget from settings.
        """
        settings = get_settings()
        self.budget_limit = budget_limit if budget_limit is not None else settings.default_budget
        self.budget_warning_threshold = settings.budget_warning_threshold

        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def get_model_pricing(self, model: str):
        """
        Get pricing info for a model.

        Args:
            model: Model name

        Returns:
            ModelPricing: Pricing information for the model
        """
        return get_model_pricing(model)

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int = 0
    ) -> float:
        """
        Calculate cost for a model given token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (default: 0)

        Returns:
            float: Cost in USD
        """
        pricing = self.get_model_pricing(model)
        return pricing.calculate_cost(input_tokens, output_tokens)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Add usage and return the cost for this call.

        This method tracks token usage and calculates the cost for a specific API call.
        It updates the internal counters for total cost and tokens.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (default: 0)

        Returns:
            float: Cost for this specific call in USD
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return cost

    def get_total_cost(self) -> float:
        """
        Get total accumulated cost.

        Returns:
            float: Total cost in USD across all usage tracked by this calculator
        """
        return self.total_cost

    def would_exceed_budget(
        self, model: str, input_tokens: int, output_tokens: int, budget_limit: Optional[float] = None
    ) -> bool:
        """
        Check if adding this usage would exceed budget.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (default: 0)
            budget_limit: Optional budget limit. If not provided, uses instance budget_limit.

        Returns:
            bool: True if the cost would exceed the budget, False otherwise
        """
        limit = budget_limit if budget_limit is not None else self.budget_limit
        additional_cost = self.calculate_cost(model, input_tokens, output_tokens)
        return (self.total_cost + additional_cost) > limit

    def is_near_budget_limit(self, threshold: Optional[float] = None) -> bool:
        """
        Check if current usage is near the budget limit.

        Args:
            threshold: Optional threshold (0-1). If not provided, uses budget_warning_threshold from settings.

        Returns:
            bool: True if usage is at or above the threshold percentage of budget
        """
        warn_threshold = threshold if threshold is not None else self.budget_warning_threshold
        return self.total_cost >= (self.budget_limit * warn_threshold)

    def get_remaining_budget(self) -> float:
        """
        Get remaining budget.

        Returns:
            float: Remaining budget in USD
        """
        return max(0, self.budget_limit - self.total_cost)

    def reset(self) -> None:
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_usage_summary(self) -> Dict[str, float]:
        """
        Get usage summary.

        Returns:
            Dict containing:
                - total_cost: Total cost in USD
                - total_input_tokens: Total input tokens
                - total_output_tokens: Total output tokens
                - total_tokens: Total tokens (input + output)
                - budget_limit: Budget limit in USD
                - remaining_budget: Remaining budget in USD
                - budget_usage_percentage: Percentage of budget used
        """
        budget_usage = (self.total_cost / self.budget_limit * 100) if self.budget_limit > 0 else 0

        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.get_remaining_budget(),
            "budget_usage_percentage": budget_usage,
        }

    def set_budget_limit(self, budget_limit: float) -> None:
        """
        Update the budget limit.

        Args:
            budget_limit: New budget limit in USD
        """
        self.budget_limit = budget_limit

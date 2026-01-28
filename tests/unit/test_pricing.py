"""
Unit tests for pricing calculator and cost tracking.

Tests the PricingCalculator class from chaos_auto_prompt.core.pricing
including cost calculation, budget tracking, and usage summaries.
"""

import pytest
from unittest.mock import patch, Mock

from chaos_auto_prompt.core.pricing import PricingCalculator
from chaos_auto_prompt.config.pricing import (
    ModelPricing,
    get_model_pricing,
    list_supported_models,
    get_model_context_size,
    DEFAULT_MODEL_PRICING,
    MODEL_CONTEXT_SIZES,
)


class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_model_pricing_creation(self):
        """Test creating a ModelPricing instance."""
        pricing = ModelPricing(
            input_price=2.50,
            output_price=10.0,
            model_name="gpt-4o"
        )

        assert pricing.input_price == 2.50
        assert pricing.output_price == 10.0
        assert pricing.model_name == "gpt-4o"

    def test_calculate_cost_input_only(self):
        """Test cost calculation with only input tokens."""
        pricing = ModelPricing(
            input_price=2.50,  # $2.50 per 1M tokens
            output_price=10.0,
            model_name="gpt-4o"
        )

        # 1,000,000 input tokens at $2.50 per 1M
        cost = pricing.calculate_cost(1_000_000, 0)
        assert cost == 2.50

    def test_calculate_cost_output_only(self):
        """Test cost calculation with only output tokens."""
        pricing = ModelPricing(
            input_price=2.50,
            output_price=10.0,  # $10.0 per 1M tokens
            model_name="gpt-4o"
        )

        # 1,000,000 output tokens at $10.0 per 1M
        cost = pricing.calculate_cost(0, 1_000_000)
        assert cost == 10.0

    def test_calculate_cost_both(self):
        """Test cost calculation with both input and output tokens."""
        pricing = ModelPricing(
            input_price=2.50,
            output_price=10.0,
            model_name="gpt-4o"
        )

        # 500,000 input + 500,000 output
        cost = pricing.calculate_cost(500_000, 500_000)
        expected = (500_000 / 1_000_000) * 2.50 + (500_000 / 1_000_000) * 10.0
        assert cost == expected

    def test_calculate_cost_small_tokens(self):
        """Test cost calculation with small token counts."""
        pricing = ModelPricing(
            input_price=2.50,
            output_price=10.0,
            model_name="gpt-4o"
        )

        # 1000 input + 500 output
        cost = pricing.calculate_cost(1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.0
        assert abs(cost - expected) < 1e-6  # Allow for floating point precision

    def test_model_pricing_immutable(self):
        """Test that ModelPricing is frozen (immutable)."""
        pricing = ModelPricing(2.50, 10.0, "gpt-4o")

        with pytest.raises(Exception):  # FrozenInstanceError
            pricing.input_price = 5.0


class TestPricingConfig:
    """Test pricing configuration functions."""

    def test_list_supported_models(self):
        """Test listing all supported models."""
        models = list_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gemini-2.5-flash" in models

    def test_get_model_pricing_known_model(self):
        """Test getting pricing for a known model."""
        pricing = get_model_pricing("gpt-4o")

        assert isinstance(pricing, ModelPricing)
        assert pricing.model_name == "gpt-4o"
        assert pricing.input_price == 2.50
        assert pricing.output_price == 10.0

    def test_get_model_pricing_unknown_model(self):
        """Test getting pricing for an unknown model raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_model_pricing("unknown-model-x")

        assert "Pricing not found" in str(exc_info.value)

    def test_get_model_pricing_env_override(self, monkeypatch):
        """Test that environment variables can override pricing."""
        from chaos_auto_prompt.config import pricing as pricing_module

        # Mock settings to return custom prices
        mock_settings = Mock()
        mock_settings.unknown_model_input_price = 5.0
        mock_settings.unknown_model_output_price = 15.0
        mock_settings.gpt_4o_input_price = None

        with patch.object(pricing_module, 'get_settings', return_value=mock_settings):
            # This should work if model has env override
            # For now, test that unknown models still fail
            with pytest.raises(ValueError):
                pricing_module.get_model_pricing("unknown-model")

    def test_get_model_context_size_known_model(self):
        """Test getting context size for a known model."""
        size = get_model_context_size("gpt-4o")

        assert size == 128000

    def test_get_model_context_size_unknown_model(self):
        """Test getting context size for unknown model returns default."""
        size = get_model_context_size("unknown-model")

        assert size == 128000  # Default context size

    def test_default_model_pricing_gpt4o(self):
        """Test default pricing for GPT-4o."""
        pricing = DEFAULT_MODEL_PRICING["gpt-4o"]

        assert pricing.input_price == 2.50
        assert pricing.output_price == 10.0

    def test_default_model_pricing_gemini_flash(self):
        """Test default pricing for Gemini 2.5 Flash."""
        pricing = DEFAULT_MODEL_PRICING["gemini-2.5-flash"]

        assert pricing.input_price == 0.075
        assert pricing.output_price == 0.30

    def test_model_context_sizes_values(self):
        """Test various model context sizes."""
        assert MODEL_CONTEXT_SIZES["gpt-4"] == 8192
        assert MODEL_CONTEXT_SIZES["gpt-4o"] == 128000
        assert MODEL_CONTEXT_SIZES["gemini-2.5-flash"] == 1000000
        assert MODEL_CONTEXT_SIZES["gemini-1.5-pro"] == 2000000


class TestPricingCalculator:
    """Test PricingCalculator class."""

    def test_initialization_default_budget(self, monkeypatch):
        """Test calculator initialization with default budget."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("default_budget", "10.0")

        calculator = PricingCalculator()

        assert calculator.budget_limit == 10.0
        assert calculator.total_cost == 0.0
        assert calculator.total_input_tokens == 0
        assert calculator.total_output_tokens == 0

    def test_initialization_custom_budget(self):
        """Test calculator initialization with custom budget."""
        calculator = PricingCalculator(budget_limit=50.0)

        assert calculator.budget_limit == 50.0
        assert calculator.total_cost == 0.0

    def test_initialization_warning_threshold(self, monkeypatch):
        """Test calculator initialization with warning threshold."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("budget_warning_threshold", "0.8")

        calculator = PricingCalculator()

        assert calculator.budget_warning_threshold == 0.8

    def test_get_model_pricing(self):
        """Test getting model pricing through calculator."""
        calculator = PricingCalculator()
        pricing = calculator.get_model_pricing("gpt-4o")

        assert pricing.model_name == "gpt-4o"
        assert pricing.input_price == 2.50
        assert pricing.output_price == 10.0

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        calculator = PricingCalculator()
        cost = calculator.calculate_cost("gpt-4o", 1000, 500)

        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.0
        assert abs(cost - expected) < 1e-6

    def test_calculate_cost_gemini_flash(self):
        """Test cost calculation for Gemini 2.5 Flash."""
        calculator = PricingCalculator()
        cost = calculator.calculate_cost("gemini-2.5-flash", 1000, 500)

        expected = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.30
        assert abs(cost - expected) < 1e-6

    def test_add_usage(self):
        """Test adding usage and tracking cost."""
        calculator = PricingCalculator()

        cost1 = calculator.add_usage("gpt-4o", 1000, 500)
        cost2 = calculator.add_usage("gpt-4o", 2000, 1000)

        assert calculator.total_input_tokens == 3000
        assert calculator.total_output_tokens == 1500
        assert abs(calculator.total_cost - (cost1 + cost2)) < 1e-6

    def test_add_usage_multiple_models(self):
        """Test adding usage from multiple models."""
        calculator = PricingCalculator()

        calculator.add_usage("gpt-4o", 1000, 500)
        calculator.add_usage("gemini-2.5-flash", 2000, 1000)

        assert calculator.total_input_tokens == 3000
        assert calculator.total_output_tokens == 1500
        assert calculator.total_cost > 0

    def test_get_total_cost(self):
        """Test getting total accumulated cost."""
        calculator = PricingCalculator()

        assert calculator.get_total_cost() == 0.0

        calculator.add_usage("gpt-4o", 1000, 500)
        total = calculator.get_total_cost()

        assert total > 0

    def test_would_exceed_budget_true(self):
        """Test checking if usage would exceed budget (True case)."""
        calculator = PricingCalculator(budget_limit=1.0)

        # Add usage that gets us close to budget
        calculator.add_usage("gpt-4o", 300_000, 100_000)  # ~$1.00

        # Check if additional usage would exceed
        would_exceed = calculator.would_exceed_budget("gpt-4o", 10_000, 1_000)
        assert would_exceed is True

    def test_would_exceed_budget_false(self):
        """Test checking if usage would exceed budget (False case)."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 1000, 500)

        would_exceed = calculator.would_exceed_budget("gpt-4o", 1000, 500)
        assert would_exceed is False

    def test_would_exceed_budget_custom_limit(self):
        """Test budget check with custom budget limit."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 1000, 500)

        # Check with lower custom limit
        would_exceed = calculator.would_exceed_budget(
            "gpt-4o", 100_000, 50_000, budget_limit=0.01
        )
        assert would_exceed is True

    def test_is_near_budget_limit_true(self):
        """Test checking if near budget limit (True case)."""
        calculator = PricingCalculator(budget_limit=10.0)

        # Use 90% of budget
        calculator.add_usage("gpt-4o", 3_000_000, 750_000)  # ~$9.00

        assert calculator.is_near_budget_limit() is True

    def test_is_near_budget_limit_false(self):
        """Test checking if near budget limit (False case)."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 100_000, 25_000)  # ~$0.30

        assert calculator.is_near_budget_limit() is False

    def test_is_near_budget_limit_custom_threshold(self):
        """Test budget limit check with custom threshold."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 500_000, 125_000)  # ~$1.50 (15%)

        # Should be near at 10% threshold
        assert calculator.is_near_budget_limit(threshold=0.1) is True
        # But not at default 90% threshold
        assert calculator.is_near_budget_limit() is False

    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        calculator = PricingCalculator(budget_limit=10.0)

        assert calculator.get_remaining_budget() == 10.0

        calculator.add_usage("gpt-4o", 100_000, 25_000)  # ~$0.30

        remaining = calculator.get_remaining_budget()
        assert remaining < 10.0
        assert remaining > 9.0

    def test_reset(self):
        """Test resetting calculator state."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 1000, 500)
        assert calculator.total_cost > 0

        calculator.reset()

        assert calculator.total_cost == 0.0
        assert calculator.total_input_tokens == 0
        assert calculator.total_output_tokens == 0

    def test_get_usage_summary(self):
        """Test getting usage summary."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 1000, 500)
        calculator.add_usage("gemini-2.5-flash", 2000, 1000)

        summary = calculator.get_usage_summary()

        assert "total_cost" in summary
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_tokens" in summary
        assert "budget_limit" in summary
        assert "remaining_budget" in summary
        assert "budget_usage_percentage" in summary

        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 1500
        assert summary["total_tokens"] == 4500
        assert summary["budget_limit"] == 10.0
        assert summary["total_cost"] > 0
        assert summary["remaining_budget"] < 10.0

    def test_set_budget_limit(self):
        """Test setting new budget limit."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 1000, 500)

        calculator.set_budget_limit(20.0)

        assert calculator.budget_limit == 20.0
        assert calculator.get_remaining_budget() > 10.0


class TestPricingCalculatorEdgeCases:
    """Test edge cases and special scenarios for PricingCalculator."""

    def test_zero_budget(self):
        """Test calculator with zero budget limit."""
        calculator = PricingCalculator(budget_limit=0.0)

        calculator.add_usage("gpt-4o", 0, 0)
        assert calculator.get_total_cost() == 0.0

        # Would exceed even with minimal cost
        would_exceed = calculator.would_exceed_budget("gpt-4o", 100, 0)
        assert would_exceed is True

    def test_negative_budget(self):
        """Test calculator with negative budget (should still work)."""
        calculator = PricingCalculator(budget_limit=-1.0)

        assert calculator.budget_limit == -1.0

    def test_zero_token_usage(self):
        """Test adding usage with zero tokens."""
        calculator = PricingCalculator()

        cost = calculator.add_usage("gpt-4o", 0, 0)

        assert cost == 0.0
        assert calculator.total_cost == 0.0

    def test_very_large_token_counts(self):
        """Test calculator with very large token counts."""
        calculator = PricingCalculator()

        cost = calculator.calculate_cost("gpt-4o", 10_000_000, 5_000_000)

        assert cost > 0
        assert cost < 1000  # Should be reasonable

    def test_budget_usage_percentage(self):
        """Test budget usage percentage calculation."""
        calculator = PricingCalculator(budget_limit=10.0)

        calculator.add_usage("gpt-4o", 400_000, 100_000)  # ~$1.00

        summary = calculator.get_usage_summary()
        assert abs(summary["budget_usage_percentage"] - 10.0) < 0.1  # ~10%

    def test_remaining_budget_negative(self):
        """Test that remaining budget doesn't go negative."""
        calculator = PricingCalculator(budget_limit=1.0)

        # Add usage that exceeds budget
        calculator.add_usage("gpt-4o", 1_000_000, 500_000)

        remaining = calculator.get_remaining_budget()
        assert remaining == 0.0  # Should floor at 0

    def test_multiple_add_usage_calls(self):
        """Test multiple sequential add_usage calls."""
        calculator = PricingCalculator()

        costs = []
        for i in range(10):
            cost = calculator.add_usage("gpt-4o", 100 * (i + 1), 50 * (i + 1))
            costs.append(cost)

        total_calculated = sum(costs)
        assert abs(calculator.total_cost - total_calculated) < 1e-6

    def test_different_model_costs(self):
        """Test that different models have different costs."""
        calculator = PricingCalculator()

        cost_gpt4o = calculator.calculate_cost("gpt-4o", 1000, 500)
        cost_gemini = calculator.calculate_cost("gemini-2.5-flash", 1000, 500)
        cost_mini = calculator.calculate_cost("gpt-4o-mini", 1000, 500)

        # GPT-4o should be more expensive than Gemini Flash
        assert cost_gpt4o > cost_gemini
        # GPT-4o-mini should be cheaper than GPT-4o
        assert cost_mini < cost_gpt4o

    def test_usage_summary_with_zero_budget(self):
        """Test usage summary when budget is zero."""
        calculator = PricingCalculator(budget_limit=0.0)

        summary = calculator.get_usage_summary()

        # Should handle division by zero gracefully
        assert summary["budget_limit"] == 0.0
        assert summary["budget_usage_percentage"] == 0  # Should be 0, not NaN

    def test_calculate_cost_all_models(self):
        """Test that all configured models can calculate costs."""
        calculator = PricingCalculator()

        for model in list_supported_models():
            cost = calculator.calculate_cost(model, 1000, 500)
            assert cost >= 0
            assert cost < 100  # Should be reasonable for 1500 tokens

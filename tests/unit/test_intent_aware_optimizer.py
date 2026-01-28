"""
Tests for IntentAwarePromptOptimizer.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from chaos_auto_prompt.optimizers.intent_aware_optimizer import IntentAwarePromptOptimizer


@pytest.mark.asyncio
class TestIntentAwarePromptOptimizer:
    """Test suite for IntentAwarePromptOptimizer."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.generate_text = AsyncMock()
        provider.generate_text_with_retry = AsyncMock()
        return provider

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        return pd.DataFrame({
            "input": [
                "Player scores 58th goal, breaking team record",
                "Team wins championship for third year",
                "Coach announces retirement",
                "Player signs new contract",
            ],
            "output": [True, True, False, False]
        })

    @pytest.fixture
    def mock_intent(self):
        """Mock intent definition."""
        return {
            "concept": "milestone",
            "definition": "A significant achievement marking record-breaking performance",
            "positive_indicators": ["record", "breaking", "championship", "historic"],
            "negative_indicators": ["retirement", "contract", "routine"],
            "boundary_cases": ["Consecutive wins only if breaking streak record"],
            "confidence": 0.9,
            "metadata": {
                "examples_analyzed": 4,
                "true_count": 2,
                "false_count": 2,
            }
        }

    async def test_init_with_intent(self, mock_provider, mock_intent):
        """Test initialization with pre-provided intent."""
        optimizer = IntentAwarePromptOptimizer(
            prompt="classify this article contain milestone or not",
            provider=mock_provider,
            intent=mock_intent,
            extract_intent=False,
        )

        assert optimizer.intent == mock_intent
        assert optimizer.extract_intent is False
        assert optimizer.intent_extractor is None

    async def test_init_with_extract_intent(self, mock_provider):
        """Test initialization with intent extraction enabled."""
        optimizer = IntentAwarePromptOptimizer(
            prompt="classify this article contain milestone or not",
            provider=mock_provider,
            model_choice="gpt-4o",
            extract_intent=True,
        )

        assert optimizer.extract_intent is True
        assert optimizer.intent_extractor is not None
        assert optimizer.intent is None  # Not extracted yet

    async def test_init_without_provider_raises_error(self):
        """Test initialization fails without provider when extract_intent=True."""
        with pytest.raises(ValueError, match="Provider required for intent extraction"):
            IntentAwarePromptOptimizer(
                prompt="classify milestone",
                extract_intent=True,
            )

    async def test_get_intent(self, mock_provider, mock_intent):
        """Test get_intent() method."""
        optimizer = IntentAwarePromptOptimizer(
            prompt="classify milestone",
            provider=mock_provider,
            intent=mock_intent,
            extract_intent=False,
        )

        assert optimizer.get_intent() == mock_intent

    async def test_optimize_with_provided_intent(
        self, mock_provider, sample_dataset, mock_intent
    ):
        """Test optimize() with pre-provided intent (skip Phase 0)."""
        # Mock meta_prompter
        mock_meta_prompter = MagicMock()
        mock_meta_prompter.construct_intent_aware_content = MagicMock(
            return_value="optimized meta-prompt"
        )

        # Mock provider response
        mock_provider.generate_text_with_retry.return_value = "Optimized prompt with intent"

        optimizer = IntentAwarePromptOptimizer(
            prompt="classify milestone",
            provider=mock_provider,
            intent=mock_intent,
            extract_intent=False,
            verbose=True,
        )
        optimizer.meta_prompter = mock_meta_prompter

        # Add feedback column to avoid validation error
        sample_dataset["feedback"] = ["good", "good", "bad", "bad"]

        result = await optimizer.optimize(
            dataset=sample_dataset,
            output_column="output",
            feedback_columns=["feedback"],
            use_two_stage_evaluator=False,
        )

        # Should use intent-aware meta-prompt
        assert mock_meta_prompter.construct_intent_aware_content.called
        assert "Optimized prompt with intent" in result

    async def test_optimize_extracts_intent_when_enabled(
        self, mock_provider, sample_dataset, mock_intent
    ):
        """Test optimize() extracts intent when enabled (Phase 0)."""
        # Mock IntentExtractor
        mock_extractor = MagicMock()
        mock_extractor.extract_intent = AsyncMock(return_value=mock_intent)

        # Mock meta_prompter
        mock_meta_prompter = MagicMock()
        mock_meta_prompter.construct_intent_aware_content = MagicMock(
            return_value="optimized meta-prompt"
        )

        # Mock provider response
        mock_provider.generate_text_with_retry.return_value = "Optimized prompt"

        optimizer = IntentAwarePromptOptimizer(
            prompt="classify milestone",
            provider=mock_provider,
            extract_intent=True,
            verbose=True,
        )
        optimizer.intent_extractor = mock_extractor
        optimizer.meta_prompter = mock_meta_prompter

        # Add feedback column to avoid validation error
        sample_dataset["feedback"] = ["good", "good", "bad", "bad"]

        result = await optimizer.optimize(
            dataset=sample_dataset,
            output_column="output",
            feedback_columns=["feedback"],
            use_two_stage_evaluator=False,
        )

        # Should extract intent
        assert mock_extractor.extract_intent.called
        assert optimizer.intent == mock_intent

        # Should use intent-aware meta-prompt
        assert mock_meta_prompter.construct_intent_aware_content.called

    async def test_optimize_fallback_without_intent(
        self, mock_provider, sample_dataset
    ):
        """Test optimize() falls back to standard meta-prompt without intent."""
        # Mock meta_prompter
        mock_meta_prompter = MagicMock()
        mock_meta_prompter.construct_content = MagicMock(
            return_value="standard meta-prompt"
        )

        # Mock provider response
        mock_provider.generate_text_with_retry.return_value = "Optimized prompt"

        optimizer = IntentAwarePromptOptimizer(
            prompt="classify milestone",
            provider=mock_provider,
            intent=None,
            extract_intent=False,
            verbose=True,
        )
        optimizer.meta_prompter = mock_meta_prompter

        # Add feedback column
        sample_dataset["feedback"] = ["good", "good", "bad", "bad"]

        result = await optimizer.optimize(
            dataset=sample_dataset,
            output_column="output",
            feedback_columns=["feedback"],
            use_two_stage_evaluator=False,
        )

        # Should use standard meta-prompt
        assert mock_meta_prompter.construct_content.called
        assert not hasattr(mock_meta_prompter, 'construct_intent_aware_content') or \
               not mock_meta_prompter.construct_intent_aware_content.called

    async def test_optimize_with_two_stage_evaluator(
        self, mock_provider, sample_dataset, mock_intent
    ):
        """Test optimize() with TwoStageEvaluator integration."""
        # Mock meta_prompter
        mock_meta_prompter = MagicMock()
        mock_meta_prompter.construct_intent_aware_content = MagicMock(
            return_value="meta-prompt"
        )

        # Mock provider response
        mock_provider.generate_text_with_retry.return_value = "Optimized prompt"

        optimizer = IntentAwarePromptOptimizer(
            prompt="classify milestone",
            provider=mock_provider,
            intent=mock_intent,
            extract_intent=False,
            verbose=True,
        )
        optimizer.meta_prompter = mock_meta_prompter

        # Mock run_evaluators to avoid actual evaluation
        async def mock_run_evaluators(dataset, evaluators, feedback_columns):
            # Add mock feedback columns
            dataset["construction_score"] = [0.8, 0.7, 0.6, 0.5]
            dataset["reasoning_score"] = [1.0, 1.0, 0.0, 0.0]
            dataset["two_stage_feedback"] = ["good", "good", "bad", "bad"]
            return dataset, ["construction_score", "reasoning_score", "two_stage_feedback"]

        optimizer.run_evaluators = mock_run_evaluators

        result = await optimizer.optimize(
            dataset=sample_dataset,
            output_column="output",
            feedback_columns=None,  # Should auto-generate
            use_two_stage_evaluator=True,
        )

        # Should have generated feedback
        assert "Optimized prompt" in result

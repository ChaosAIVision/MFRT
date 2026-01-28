"""
Tests for IntentExtractor.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from chaos_auto_prompt.utils.intent_extractor import IntentExtractor


@pytest.mark.asyncio
class TestIntentExtractor:
    """Test suite for IntentExtractor."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.generate_text = AsyncMock()
        return provider

    @pytest.fixture
    def sample_examples(self):
        """Sample labeled examples."""
        return pd.DataFrame({
            "input": [
                "Player scores 58th goal, breaking team record",
                "Team wins championship for third year",
                "Coach announces retirement",
                "Player signs new contract",
                "Stadium renovation begins",
                "Historic 100th goal scored",
            ],
            "output": [True, True, False, False, False, True]
        })

    @pytest.fixture
    def mock_intent_response(self):
        """Mock LLM response for intent extraction."""
        return """<intent>
  <concept>milestone</concept>
  <definition>A significant sports achievement that marks record-breaking performance or historic numeric milestones (e.g., 50th, 100th) in player or team statistics.</definition>
  <positive_indicators>
    <indicator>record-breaking</indicator>
    <indicator>historic numeric milestone (50th, 100th, 1000th)</indicator>
    <indicator>championship victory</indicator>
    <indicator>first time achievement</indicator>
  </positive_indicators>
  <negative_indicators>
    <indicator>routine contract signing</indicator>
    <indicator>retirement announcement</indicator>
    <indicator>facility renovation</indicator>
    <indicator>regular game without records</indicator>
  </negative_indicators>
  <boundary_cases>
    <case>Consecutive wins only count if breaking a streak record</case>
    <case>High scores only count if personal or team record</case>
  </boundary_cases>
  <confidence>0.9</confidence>
</intent>"""

    async def test_extract_intent_success(self, mock_provider, sample_examples, mock_intent_response):
        """Test successful intent extraction."""
        mock_provider.generate_text.return_value = mock_intent_response

        extractor = IntentExtractor(provider=mock_provider)
        intent = await extractor.extract_intent(
            system_prompt="classify this article contain milestone or not",
            examples=sample_examples,
        )

        assert intent["concept"] == "milestone"
        assert "record-breaking" in intent["definition"]
        assert len(intent["positive_indicators"]) == 4
        assert len(intent["negative_indicators"]) == 4
        assert len(intent["boundary_cases"]) == 2
        assert intent["confidence"] == 0.9
        assert intent["metadata"]["examples_analyzed"] <= 10

    async def test_extract_concept_name(self, mock_provider):
        """Test concept name extraction from system prompt."""
        extractor = IntentExtractor(provider=mock_provider)

        test_cases = [
            ("classify this article contain milestone or not", "milestone"),
            ("determine if email is spam", "spam"),
            ("check if task is urgent", "urgent"),
            ("contains sensitive information", "sensitive"),
        ]

        for prompt, expected_concept in test_cases:
            concept = extractor._extract_concept_name(prompt)
            assert concept == expected_concept

    async def test_sample_balanced_examples(self, mock_provider, sample_examples):
        """Test balanced sampling of examples."""
        extractor = IntentExtractor(provider=mock_provider)

        sampled = extractor._sample_balanced_examples(
            sample_examples, "output", max_examples=4
        )

        assert len(sampled) == 4
        # Should have 2 True and 2 False (balanced)
        true_count = sampled["output"].sum()
        assert true_count == 2

    async def test_fallback_intent(self, mock_provider, sample_examples):
        """Test fallback intent when extraction fails."""
        mock_provider.generate_text.side_effect = Exception("LLM error")

        extractor = IntentExtractor(provider=mock_provider)
        intent = await extractor.extract_intent(
            system_prompt="classify milestone",
            examples=sample_examples,
        )

        # Should return fallback intent
        assert intent["concept"] == "milestone"
        assert intent["confidence"] < 0.5
        assert "fallback" in intent.get("metadata", {})

    async def test_invalid_inputs(self, mock_provider, sample_examples):
        """Test validation of invalid inputs."""
        extractor = IntentExtractor(provider=mock_provider)

        # Empty system prompt
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            await extractor.extract_intent(
                system_prompt="",
                examples=sample_examples,
            )

        # Empty examples
        with pytest.raises(ValueError, match="examples cannot be empty"):
            await extractor.extract_intent(
                system_prompt="classify milestone",
                examples=pd.DataFrame(),
            )

        # Missing output column
        with pytest.raises(ValueError, match="output_column.*not found"):
            await extractor.extract_intent(
                system_prompt="classify milestone",
                examples=sample_examples,
                output_column="wrong_column",
            )

    async def test_parse_intent_response(self, mock_provider, mock_intent_response):
        """Test parsing of LLM response."""
        extractor = IntentExtractor(provider=mock_provider)

        intent = extractor._parse_intent_response(mock_intent_response, "fallback_concept")

        assert intent["concept"] == "milestone"
        assert "significant sports achievement" in intent["definition"]
        assert "record-breaking" in intent["positive_indicators"]
        assert "routine contract signing" in intent["negative_indicators"]
        assert "Consecutive wins" in intent["boundary_cases"][0]
        assert intent["confidence"] == 0.9

    async def test_parse_malformed_response(self, mock_provider):
        """Test parsing malformed LLM response."""
        extractor = IntentExtractor(provider=mock_provider)

        malformed_response = "This is not valid XML"
        intent = extractor._parse_intent_response(malformed_response, "milestone")

        # Should return fallback with concept
        assert intent["concept"] == "milestone"
        assert intent["confidence"] < 1.0

    async def test_max_examples_limit(self, mock_provider, mock_intent_response):
        """Test max_examples parameter."""
        mock_provider.generate_text.return_value = mock_intent_response

        # Create larger dataset
        large_dataset = pd.DataFrame({
            "input": [f"Article {i}" for i in range(50)],
            "output": [i % 2 == 0 for i in range(50)]
        })

        extractor = IntentExtractor(provider=mock_provider)
        intent = await extractor.extract_intent(
            system_prompt="classify milestone",
            examples=large_dataset,
            max_examples=6,
        )

        # Should analyze only 6 examples
        assert intent["metadata"]["examples_analyzed"] == 6

    async def test_unbalanced_dataset(self, mock_provider, mock_intent_response):
        """Test with unbalanced dataset."""
        mock_provider.generate_text.return_value = mock_intent_response

        # Create heavily imbalanced dataset (9 True, 1 False)
        unbalanced = pd.DataFrame({
            "input": [f"Article {i}" for i in range(10)],
            "output": [True] * 9 + [False]
        })

        extractor = IntentExtractor(provider=mock_provider)
        sampled = extractor._sample_balanced_examples(unbalanced, "output", max_examples=6)

        # Should still sample both classes
        true_count = sampled["output"].sum()
        false_count = len(sampled) - true_count
        assert true_count > 0
        assert false_count > 0

    async def test_metadata_included(self, mock_provider, sample_examples, mock_intent_response):
        """Test that metadata is included in result."""
        mock_provider.generate_text.return_value = mock_intent_response

        extractor = IntentExtractor(provider=mock_provider)
        intent = await extractor.extract_intent(
            system_prompt="classify milestone",
            examples=sample_examples,
        )

        assert "metadata" in intent
        assert "examples_analyzed" in intent["metadata"]
        assert "true_count" in intent["metadata"]
        assert "false_count" in intent["metadata"]
        assert "system_prompt" in intent["metadata"]

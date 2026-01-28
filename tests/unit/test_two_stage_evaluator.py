"""
Tests for TwoStageEvaluator.
"""

import pytest
import pandas as pd
from chaos_auto_prompt.evaluators.two_stage import TwoStageEvaluator


@pytest.mark.asyncio
class TestTwoStageEvaluator:
    """Test suite for TwoStageEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a TwoStageEvaluator instance."""
        return TwoStageEvaluator(
            feedback_column="two_stage_feedback",
            construction_weight=0.5,
            reasoning_weight=0.5,
            groundtruth_column="output",
            prediction_column="prediction",
        )

    @pytest.fixture
    def sample_construction(self):
        """Sample complete construction text."""
        return """
        (1) Relevant Entities: Article, Milestone, Record, Achievement
        (2) State Variables: article_content: text, contains_milestone: boolean
        (3) Possible Actions: classify_as_milestone, Preconditions: mentions record/achievement
        (4) Constraints: Binary output only (True/False)
        """

    @pytest.fixture
    def sample_reasoning(self):
        """Sample reasoning text."""
        return """
        Step 1: Identify milestone indicators
        - Article mentions "record", "first time", "historic"

        Step 2: Apply classification rules
        - Multiple milestone indicators present

        Step 3: Final decision
        - Classification: True
        - Confidence: high
        """

    async def test_evaluate_correct_prediction(self, evaluator):
        """Test evaluation with correct prediction."""
        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
        })

        result_df, feedback_cols = await evaluator.evaluate(df)

        assert "construction_score" in result_df.columns
        assert "reasoning_score" in result_df.columns
        assert "overall_correct" in result_df.columns
        assert result_df["overall_correct"].iloc[0] == True
        assert result_df["reasoning_score"].iloc[0] == 1.0

    async def test_evaluate_incorrect_prediction(self, evaluator):
        """Test evaluation with incorrect prediction."""
        df = pd.DataFrame({
            "prediction": ["False"],
            "output": ["True"],
        })

        result_df, feedback_cols = await evaluator.evaluate(df)

        assert result_df["overall_correct"].iloc[0] == False
        assert result_df["reasoning_score"].iloc[0] == 0.0

    async def test_evaluate_with_construction(self, evaluator, sample_construction):
        """Test evaluation with construction text."""
        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
            "construction": [sample_construction],
        })

        evaluator.construction_column = "construction"
        result_df, feedback_cols = await evaluator.evaluate(df)

        # Should have high construction score (all 4 sections present)
        assert result_df["construction_score"].iloc[0] == 1.0
        assert "entities" in result_df["construction_feedback"].iloc[0].lower()

    async def test_evaluate_partial_construction(self, evaluator):
        """Test evaluation with incomplete construction."""
        partial_construction = """
        (1) Relevant Entities: Article, Milestone
        (2) State Variables: article_content: text
        """

        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
            "construction": [partial_construction],
        })

        evaluator.construction_column = "construction"
        result_df, feedback_cols = await evaluator.evaluate(df)

        # Should have 0.5 score (2/4 sections)
        assert result_df["construction_score"].iloc[0] == 0.5
        assert "missing" in result_df["construction_feedback"].iloc[0].lower()

    async def test_evaluate_no_construction(self, evaluator):
        """Test evaluation when no construction provided."""
        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
        })

        result_df, feedback_cols = await evaluator.evaluate(df)

        # Should have 0.0 construction score
        assert result_df["construction_score"].iloc[0] == 0.0
        assert "no construction" in result_df["construction_feedback"].iloc[0].lower()

    async def test_evaluate_with_reasoning(self, evaluator, sample_reasoning):
        """Test evaluation with reasoning text."""
        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
            "reasoning": [sample_reasoning],
        })

        evaluator.reasoning_column = "reasoning"
        result_df, feedback_cols = await evaluator.evaluate(df)

        assert result_df["reasoning_score"].iloc[0] == 1.0
        assert "structured reasoning" in result_df["reasoning_feedback"].iloc[0].lower()

    async def test_normalize_value(self, evaluator):
        """Test value normalization."""
        assert evaluator._normalize_value("True") == "true"
        assert evaluator._normalize_value("TRUE") == "true"
        assert evaluator._normalize_value("true") == "true"
        assert evaluator._normalize_value("Yes") == "true"
        assert evaluator._normalize_value("1") == "true"

        assert evaluator._normalize_value("False") == "false"
        assert evaluator._normalize_value("FALSE") == "false"
        assert evaluator._normalize_value("false") == "false"
        assert evaluator._normalize_value("No") == "false"
        assert evaluator._normalize_value("0") == "false"

    async def test_overall_score_calculation(self, evaluator, sample_construction):
        """Test overall score is weighted average."""
        df = pd.DataFrame({
            "prediction": ["True"],
            "output": ["True"],
            "construction": [sample_construction],
        })

        evaluator.construction_column = "construction"
        result_df, feedback_cols = await evaluator.evaluate(df)

        # Construction: 1.0, Reasoning: 1.0
        # Overall = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert result_df["overall_score"].iloc[0] == 1.0

    async def test_weight_normalization(self):
        """Test weights are normalized if they don't sum to 1.0."""
        evaluator = TwoStageEvaluator(
            construction_weight=0.6,
            reasoning_weight=0.6,  # Intentionally wrong
        )

        # Weights should be normalized to 0.5, 0.5
        assert abs(evaluator.construction_weight - 0.5) < 0.001
        assert abs(evaluator.reasoning_weight - 0.5) < 0.001

    async def test_feedback_columns_list(self, evaluator):
        """Test get_feedback_columns returns all expected columns."""
        columns = evaluator.get_feedback_columns()

        assert "construction_score" in columns
        assert "construction_feedback" in columns
        assert "reasoning_score" in columns
        assert "reasoning_feedback" in columns
        assert "overall_score" in columns
        assert "overall_correct" in columns
        assert "two_stage_feedback" in columns

    async def test_batch_evaluation(self, evaluator):
        """Test evaluation works with multiple rows."""
        df = pd.DataFrame({
            "prediction": ["True", "False", "True"],
            "output": ["True", "True", "True"],
        })

        result_df, feedback_cols = await evaluator.evaluate(df)

        assert len(result_df) == 3
        assert result_df["overall_correct"].tolist() == [True, False, True]
        assert result_df["reasoning_score"].tolist() == [1.0, 0.0, 1.0]

"""
Two-Stage Evaluator - Separate evaluation for Construction and Reasoning phases.

This evaluator provides granular feedback by scoring construction quality
and reasoning correctness independently, enabling better optimization insights.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ..interfaces.evaluator import BaseEvaluator
from ..utils.construction_extractor import ConstructionExtractor

logger = logging.getLogger(__name__)


class TwoStageEvaluator(BaseEvaluator):
    """
    Evaluator that separately scores construction quality and reasoning correctness.

    Stage 1 (Construction): Evaluates completeness of problem decomposition
        - Entities: Are all relevant objects identified?
        - State Variables: Are key attributes defined?
        - Actions: Are operations with preconditions specified?
        - Constraints: Are rules and limitations stated?

    Stage 2 (Reasoning): Evaluates correctness of reasoning path
        - Does the reasoning lead to the correct conclusion?
        - Is the final prediction accurate?

    This two-stage approach helps identify whether optimization should focus on
    improving problem understanding (construction) or logical reasoning.
    """

    def __init__(
        self,
        feedback_column: str = "two_stage_feedback",
        construction_weight: float = 0.5,
        reasoning_weight: float = 0.5,
        groundtruth_column: str = "output",
        prediction_column: str = "prediction",
        construction_column: Optional[str] = None,
        reasoning_column: Optional[str] = None,
    ):
        """
        Initialize two-stage evaluator.

        Args:
            feedback_column: Primary feedback column name
            construction_weight: Weight for construction score (0-1)
            reasoning_weight: Weight for reasoning score (0-1)
            groundtruth_column: Column containing ground truth labels
            prediction_column: Column containing model predictions
            construction_column: Optional column with construction XML
            reasoning_column: Optional column with reasoning XML
        """
        super().__init__(feedback_column)
        self.construction_weight = construction_weight
        self.reasoning_weight = reasoning_weight
        self.groundtruth_column = groundtruth_column
        self.prediction_column = prediction_column
        self.construction_column = construction_column
        self.reasoning_column = reasoning_column

        # Validate weights sum to 1.0
        if not abs((construction_weight + reasoning_weight) - 1.0) < 0.001:
            logger.warning(
                f"Weights don't sum to 1.0 ({construction_weight + reasoning_weight}). "
                "Normalizing..."
            )
            total = construction_weight + reasoning_weight
            self.construction_weight = construction_weight / total
            self.reasoning_weight = reasoning_weight / total

    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate dataframe with 2-stage scoring.

        Args:
            dataframe: DataFrame with predictions and optionally construction/reasoning
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
        """
        df = dataframe.copy()

        # Initialize feedback columns
        construction_scores = []
        reasoning_scores = []
        overall_scores = []
        construction_feedbacks = []
        reasoning_feedbacks = []
        overall_correct_list = []

        for idx, row in df.iterrows():
            # Extract data
            prediction = str(row.get(self.prediction_column, ""))
            groundtruth = str(row.get(self.groundtruth_column, ""))
            construction_text = str(row.get(self.construction_column, "")) if self.construction_column else None
            reasoning_text = str(row.get(self.reasoning_column, "")) if self.reasoning_column else None

            # Evaluate construction
            construction_result = self.evaluate_construction(construction_text)
            construction_scores.append(construction_result["score"])
            construction_feedbacks.append(construction_result["feedback"])

            # Evaluate reasoning
            reasoning_result = self.evaluate_reasoning(
                reasoning_text, prediction, groundtruth
            )
            reasoning_scores.append(reasoning_result["score"])
            reasoning_feedbacks.append(reasoning_result["feedback"])

            # Overall score
            overall_score = (
                self.construction_weight * construction_result["score"] +
                self.reasoning_weight * reasoning_result["score"]
            )
            overall_scores.append(overall_score)

            # Overall correctness (based on prediction match)
            overall_correct = self._normalize_value(prediction) == self._normalize_value(groundtruth)
            overall_correct_list.append(overall_correct)

        # Add feedback columns
        df["construction_score"] = construction_scores
        df["construction_feedback"] = construction_feedbacks
        df["reasoning_score"] = reasoning_scores
        df["reasoning_feedback"] = reasoning_feedbacks
        df["overall_score"] = overall_scores
        df["overall_correct"] = overall_correct_list
        df[self.feedback_column] = df.apply(
            lambda row: self._format_combined_feedback(row), axis=1
        )

        feedback_columns = [
            "construction_score",
            "construction_feedback",
            "reasoning_score",
            "reasoning_feedback",
            "overall_score",
            "overall_correct",
            self.feedback_column,
        ]

        return df, feedback_columns

    def evaluate_construction(self, construction_text: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate construction quality based on completeness.

        Args:
            construction_text: Extracted construction from Phase 1

        Returns:
            Dict with:
                - score: 0-1 score based on completeness
                - feedback: Detailed feedback string
                - metadata: Counts of each element found
        """
        if not construction_text or construction_text.strip() == "":
            return {
                "score": 0.0,
                "feedback": "No construction provided",
                "metadata": {
                    "entities_count": 0,
                    "state_variables_count": 0,
                    "actions_count": 0,
                    "constraints_count": 0,
                }
            }

        try:
            # Extract construction elements
            extracted = ConstructionExtractor.extract(construction_text)

            # Count elements
            entities_count = len(extracted.get("entities", []))
            state_vars_count = len(extracted.get("state_variables", []))
            actions_count = len(extracted.get("actions", []))
            constraints_count = len(extracted.get("constraints", []))

            # Score based on presence of each section (0.25 per section)
            sections_present = sum([
                1 if entities_count > 0 else 0,
                1 if state_vars_count > 0 else 0,
                1 if actions_count > 0 else 0,
                1 if constraints_count > 0 else 0,
            ])
            score = sections_present / 4.0

            # Generate feedback
            missing_sections = []
            if entities_count == 0:
                missing_sections.append("entities")
            if state_vars_count == 0:
                missing_sections.append("state variables")
            if actions_count == 0:
                missing_sections.append("actions")
            if constraints_count == 0:
                missing_sections.append("constraints")

            if missing_sections:
                feedback = f"Construction incomplete. Missing: {', '.join(missing_sections)}"
            else:
                feedback = f"Construction complete. Found {entities_count} entities, {state_vars_count} state vars, {actions_count} actions, {constraints_count} constraints"

            return {
                "score": score,
                "feedback": feedback,
                "metadata": {
                    "entities_count": entities_count,
                    "state_variables_count": state_vars_count,
                    "actions_count": actions_count,
                    "constraints_count": constraints_count,
                    "extraction_confidence": extracted.get("metadata", {}).get("extraction_confidence", 0.0),
                }
            }

        except Exception as e:
            logger.warning(f"Construction extraction failed: {e}")
            return {
                "score": 0.0,
                "feedback": f"Construction extraction error: {str(e)}",
                "metadata": {
                    "entities_count": 0,
                    "state_variables_count": 0,
                    "actions_count": 0,
                    "constraints_count": 0,
                }
            }

    def evaluate_reasoning(
        self,
        reasoning_text: Optional[str],
        prediction: str,
        groundtruth: str
    ) -> Dict[str, Any]:
        """
        Evaluate reasoning correctness based on final prediction.

        Args:
            reasoning_text: Extracted reasoning from Phase 2
            prediction: Model's final prediction
            groundtruth: Expected correct answer

        Returns:
            Dict with:
                - score: 1.0 if correct, 0.0 if incorrect
                - feedback: Explanation of correctness
        """
        # Normalize values for comparison
        pred_normalized = self._normalize_value(prediction)
        truth_normalized = self._normalize_value(groundtruth)

        is_correct = pred_normalized == truth_normalized
        score = 1.0 if is_correct else 0.0

        # Generate feedback
        if is_correct:
            feedback = f"Reasoning correct. Predicted: {prediction}, Expected: {groundtruth}"
        else:
            feedback = f"Reasoning incorrect. Predicted: {prediction}, Expected: {groundtruth}"

        # If reasoning text provided, check for logical flow markers
        if reasoning_text and reasoning_text.strip():
            has_step1 = bool(re.search(r'step\s*1', reasoning_text, re.IGNORECASE))
            has_step2 = bool(re.search(r'step\s*2', reasoning_text, re.IGNORECASE))
            has_final = bool(re.search(r'final|conclusion|decision', reasoning_text, re.IGNORECASE))

            if has_step1 and has_step2 and has_final:
                feedback += " (Structured reasoning flow detected)"
            else:
                feedback += " (Reasoning may lack clear step structure)"

        return {
            "score": score,
            "feedback": feedback,
        }

    def _normalize_value(self, value: str) -> str:
        """
        Normalize value for comparison (handles True/False, yes/no, 1/0, etc.).

        Args:
            value: String value to normalize

        Returns:
            Normalized lowercase string
        """
        value_str = str(value).strip().lower()

        # Boolean mappings
        if value_str in ["true", "yes", "1", "correct"]:
            return "true"
        if value_str in ["false", "no", "0", "incorrect"]:
            return "false"

        return value_str

    def _format_combined_feedback(self, row: pd.Series) -> str:
        """
        Format combined feedback from both stages.

        Args:
            row: DataFrame row with all feedback columns

        Returns:
            Formatted feedback string
        """
        construction_score = row.get("construction_score", 0.0)
        reasoning_score = row.get("reasoning_score", 0.0)
        overall_score = row.get("overall_score", 0.0)

        feedback_parts = [
            f"Overall Score: {overall_score:.2f}",
            f"Construction: {construction_score:.2f} - {row.get('construction_feedback', 'N/A')}",
            f"Reasoning: {reasoning_score:.2f} - {row.get('reasoning_feedback', 'N/A')}",
        ]

        return " | ".join(feedback_parts)

    def get_feedback_columns(self) -> List[str]:
        """
        Get list of all feedback columns this evaluator generates.

        Returns:
            List of column names
        """
        return [
            "construction_score",
            "construction_feedback",
            "reasoning_score",
            "reasoning_feedback",
            "overall_score",
            "overall_correct",
            self.feedback_column,
        ]


__all__ = ["TwoStageEvaluator"]

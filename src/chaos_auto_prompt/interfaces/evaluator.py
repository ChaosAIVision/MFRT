"""
Base evaluator interface for automatic feedback generation.

Evaluators analyze model outputs and generate feedback columns automatically,
enabling LLM-as-a-Judge pattern for prompt optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.

    Evaluators take a DataFrame with model outputs and generate feedback columns
    (e.g., correctness, explanation, rule_violations) that can be used for
    prompt optimization.
    """

    def __init__(self, feedback_column: str):
        """
        Initialize evaluator.

        Args:
            feedback_column: Name of the primary feedback column to generate
        """
        self.feedback_column = feedback_column

    @abstractmethod
    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate the dataframe and generate feedback columns.

        Args:
            dataframe: DataFrame containing model outputs to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
            - updated_dataframe: Original dataframe with new feedback columns added
            - feedback_column_names: List of feedback column names that were added

        Raises:
            Exception: If evaluation fails
        """
        pass

    def get_feedback_columns(self) -> List[str]:
        """
        Get list of feedback columns this evaluator generates.

        Returns:
            List of column names
        """
        return [self.feedback_column]


class LLMEvaluator(BaseEvaluator):
    """
    Base class for LLM-based evaluators.

    Uses an LLM to evaluate outputs and generate structured feedback.
    """

    def __init__(
        self,
        feedback_column: str,
        model: str,
        prompt_template: str,
        api_key: Optional[str] = None,
        provider: str = "openai",
    ):
        """
        Initialize LLM evaluator.

        Args:
            feedback_column: Primary feedback column name
            model: LLM model to use for evaluation
            prompt_template: Template for evaluation prompt
            api_key: Optional API key for LLM provider
            provider: LLM provider (openai, google)
        """
        super().__init__(feedback_column)
        self.model = model
        self.prompt_template = prompt_template
        self.api_key = api_key
        self.provider = provider


__all__ = ["BaseEvaluator", "LLMEvaluator"]

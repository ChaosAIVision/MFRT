"""
Classification evaluator for binary/multi-class evaluation.

Uses LLM to evaluate outputs and classify them into categories
(e.g., correct/incorrect) with explanations.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from chaos_auto_prompt.interfaces.evaluator import LLMEvaluator
from chaos_auto_prompt.config import get_settings


class ClassificationEvaluator(LLMEvaluator):
    """
    Evaluator that classifies outputs into categories using LLM.

    Example usage:
        ```python
        evaluator = ClassificationEvaluator(
            feedback_column="correctness",
            model="gpt-4o",
            prompt_template=\"\"\"
                Evaluate this output: {output}
                Is it correct? Return JSON with:
                "correctness": "correct" or "incorrect"
                "explanation": "your reasoning"
            \"\"\",
            choices={"correct": 1, "incorrect": 0}
        )
        ```
    """

    def __init__(
        self,
        feedback_column: str,
        model: str,
        prompt_template: str,
        choices: Dict[str, int],
        api_key: Optional[str] = None,
        provider: str = "openai",
        include_explanation: bool = True,
    ):
        """
        Initialize classification evaluator.

        Args:
            feedback_column: Name of the feedback column (e.g., "correctness")
            model: LLM model to use
            prompt_template: Template with placeholders for evaluation
            choices: Mapping of labels to scores (e.g., {"correct": 1, "incorrect": 0})
            api_key: Optional API key
            provider: LLM provider (openai, google)
            include_explanation: Whether to extract explanation field
        """
        super().__init__(feedback_column, model, prompt_template, api_key, provider)
        self.choices = choices
        self.include_explanation = include_explanation
        self.settings = get_settings()

    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        concurrency: int = 20,
        show_progress: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate dataframe using LLM classification with concurrent API calls.

        Args:
            dataframe: DataFrame with outputs to evaluate
            concurrency: Number of concurrent API calls (default: 20)
            show_progress: Show progress bar (default: True)
            **kwargs: Additional parameters for prompt template

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
        """
        # Import provider based on configuration
        if self.provider == "openai":
            from chaos_auto_prompt.providers.openai import OpenAIProvider
            provider = OpenAIProvider(
                api_key=self.api_key or self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                max_retries=self.settings.openai_max_retries,
            )
        elif self.provider == "google":
            from chaos_auto_prompt.providers.google import GoogleProvider
            provider = GoogleProvider(
                api_key=self.api_key or self.settings.google_api_key,
                timeout=self.settings.google_timeout,
                max_retries=self.settings.google_max_retries,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_row(idx: int, row: pd.Series) -> Dict[str, Any]:
            """Evaluate a single row with semaphore control."""
            async with semaphore:
                try:
                    # Format prompt with row data
                    prompt = self.prompt_template.format(**row.to_dict(), **kwargs)

                    # Generate evaluation
                    messages = [{"role": "user", "content": prompt}]
                    response = await provider.generate_text_with_retry(
                        messages=messages,
                        model=self.model,
                    )

                    # Parse response
                    return self._parse_response(response)

                except Exception as e:
                    print(f"Error evaluating row {idx}: {e}")
                    return {
                        self.feedback_column: None,
                        "explanation": f"Error: {str(e)}"
                    }

        # Create tasks for all rows
        tasks = [evaluate_row(idx, row) for idx, row in dataframe.iterrows()]

        # Execute with progress bar (like phoenix.evals)
        if show_progress:
            results = await async_tqdm.gather(
                *tasks,
                desc="Evaluating",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )
        else:
            results = await asyncio.gather(*tasks)

        # Add results to dataframe
        df_copy = dataframe.copy()

        for col_name in [self.feedback_column, "explanation"]:
            if col_name == "explanation" and not self.include_explanation:
                continue
            df_copy[col_name] = [r.get(col_name) for r in results]

        # Determine feedback columns
        feedback_columns = [self.feedback_column]
        if self.include_explanation:
            feedback_columns.append("explanation")

        return df_copy, feedback_columns

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract classification and explanation.

        Args:
            response: LLM response text

        Returns:
            Dictionary with feedback_column and explanation
        """
        result = {
            self.feedback_column: None,
            "explanation": None,
        }

        try:
            # Try to parse as JSON first
            data = json.loads(response)
            result[self.feedback_column] = data.get(self.feedback_column)
            result["explanation"] = data.get("explanation")
        except json.JSONDecodeError:
            # Fallback to regex parsing
            result[self.feedback_column] = self._extract_classification(response)
            result["explanation"] = self._extract_explanation(response)

        return result

    def _extract_classification(self, text: str) -> Optional[str]:
        """Extract classification label from text."""
        # Look for the feedback column in quotes
        pattern = rf'"{self.feedback_column}":\s*"?([\w]+)"?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label = match.group(1).lower()
            # Find matching choice
            for choice in self.choices.keys():
                if choice.lower() == label:
                    return choice
        return None

    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation from text."""
        # Look for explanation field
        pattern = r'"explanation":\s*"([^"]*)"'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def get_feedback_columns(self) -> List[str]:
        """Get list of feedback columns this evaluator generates."""
        columns = [self.feedback_column]
        if self.include_explanation:
            columns.append("explanation")
        return columns


__all__ = ["ClassificationEvaluator"]

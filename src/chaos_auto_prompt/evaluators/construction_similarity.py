"""
Construction Similarity Evaluator for Phase 1 optimization.

This evaluator compares model construction outputs with ground truth constructions
to ensure the model can correctly identify:
(1) relevant entities,
(2) state variables,
(3) possible actions with preconditions and effects,
and (4) constraints.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from chaos_auto_prompt.interfaces.evaluator import LLMEvaluator
from chaos_auto_prompt.config import get_settings


class ConstructionSimilarityEvaluator(LLMEvaluator):
    """
    Evaluator that measures similarity between generated construction and ground truth.

    Compares the four key components:
    - Entities
    - State variables
    - Actions (with preconditions and effects)
    - Constraints

    Target: 90% similarity with ground truth construction.

    Example usage:
        ```python
        evaluator = ConstructionSimilarityEvaluator(
            feedback_column="construction_similarity",
            model="gpt-4o",
            similarity_threshold=0.9
        )

        # Dataset should have columns:
        # - construction: model's construction output in <construction> tags
        # - ground_truth_construction: correct construction

        df, feedback_cols = await evaluator.evaluate(dataset)
        ```
    """

    def __init__(
        self,
        feedback_column: str = "construction_similarity",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        provider: str = "openai",
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize construction similarity evaluator.

        Args:
            feedback_column: Name of the feedback column (default: "construction_similarity")
            model: LLM model to use for evaluation
            api_key: Optional API key
            provider: LLM provider (openai, google)
            similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.9)
        """
        prompt_template = """You are an expert evaluator for prompt optimization.

Compare the MODEL CONSTRUCTION with the GROUND TRUTH CONSTRUCTION.

MODEL CONSTRUCTION:
{construction}

GROUND TRUTH CONSTRUCTION:
{ground_truth_construction}

Evaluate how similar they are across these four components:
1. Entities: Are the relevant entities correctly identified?
2. State Variables: Are the state variables properly defined?
3. Actions: Are the actions with preconditions and effects accurate?
4. Constraints: Are the constraints correctly specified?

Return a JSON with:
{{
  "construction_similarity": <float between 0.0 and 1.0>,
  "entities_score": <float between 0.0 and 1.0>,
  "state_variables_score": <float between 0.0 and 1.0>,
  "actions_score": <float between 0.0 and 1.0>,
  "constraints_score": <float between 0.0 and 1.0>,
  "explanation": "<detailed explanation of similarities and differences>",
  "passes_threshold": <true if construction_similarity >= {threshold}, else false>
}}

The construction_similarity should be the average of the four component scores.
"""

        super().__init__(
            feedback_column=feedback_column,
            model=model,
            prompt_template=prompt_template,
            api_key=api_key,
            provider=provider
        )
        self.similarity_threshold = similarity_threshold
        self.settings = get_settings()

    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        concurrency: int = 20,
        show_progress: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate construction similarity for all rows.

        Args:
            dataframe: DataFrame with 'construction' and 'ground_truth_construction' columns
            concurrency: Number of concurrent API calls (default: 20)
            show_progress: Show progress bar (default: True)
            **kwargs: Additional parameters

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
        """
        # Validate required columns
        required_columns = ['construction', 'ground_truth_construction']
        missing = [col for col in required_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Import provider
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
                    # Extract construction content from tags if present
                    construction = self._extract_construction(row['construction'])
                    ground_truth = self._extract_construction(row['ground_truth_construction'])

                    # Format prompt
                    prompt = self.prompt_template.format(
                        construction=construction,
                        ground_truth_construction=ground_truth,
                        threshold=self.similarity_threshold,
                        **kwargs
                    )

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
                        self.feedback_column: 0.0,
                        "entities_score": 0.0,
                        "state_variables_score": 0.0,
                        "actions_score": 0.0,
                        "constraints_score": 0.0,
                        "explanation": f"Error: {str(e)}",
                        "passes_threshold": False
                    }

        # Create tasks for all rows
        tasks = [evaluate_row(idx, row) for idx, row in dataframe.iterrows()]

        # Execute with progress bar
        if show_progress:
            results = await async_tqdm.gather(
                *tasks,
                desc="Evaluating Construction Similarity",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )
        else:
            results = await asyncio.gather(*tasks)

        # Add results to dataframe
        df_copy = dataframe.copy()

        feedback_columns = [
            self.feedback_column,
            "entities_score",
            "state_variables_score",
            "actions_score",
            "constraints_score",
            "explanation",
            "passes_threshold"
        ]

        for col_name in feedback_columns:
            df_copy[col_name] = [r.get(col_name) for r in results]

        return df_copy, feedback_columns

    def _extract_construction(self, text: str) -> str:
        """
        Extract construction content from <construction> tags if present.

        Args:
            text: Text potentially containing <construction> tags

        Returns:
            Extracted construction content or original text
        """
        if not isinstance(text, str):
            return str(text)

        # Try to extract from <construction></construction> tags
        match = re.search(r'<construction>(.*?)</construction>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return text.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract similarity metrics.

        Args:
            response: LLM response text

        Returns:
            Dictionary with similarity scores and explanation
        """
        result = {
            self.feedback_column: 0.0,
            "entities_score": 0.0,
            "state_variables_score": 0.0,
            "actions_score": 0.0,
            "constraints_score": 0.0,
            "explanation": None,
            "passes_threshold": False
        }

        try:
            # Try to parse as JSON
            # First try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else response

            data = json.loads(json_str)

            result[self.feedback_column] = float(data.get("construction_similarity", 0.0))
            result["entities_score"] = float(data.get("entities_score", 0.0))
            result["state_variables_score"] = float(data.get("state_variables_score", 0.0))
            result["actions_score"] = float(data.get("actions_score", 0.0))
            result["constraints_score"] = float(data.get("constraints_score", 0.0))
            result["explanation"] = data.get("explanation", "")
            result["passes_threshold"] = bool(data.get("passes_threshold", False))

        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            # Fallback: try to extract numbers from text
            result["explanation"] = f"Parse error: {str(e)}. Response: {response[:200]}"

        return result

    def get_feedback_columns(self) -> List[str]:
        """Get list of feedback columns this evaluator generates."""
        return [
            self.feedback_column,
            "entities_score",
            "state_variables_score",
            "actions_score",
            "constraints_score",
            "explanation",
            "passes_threshold"
        ]


__all__ = ["ConstructionSimilarityEvaluator"]

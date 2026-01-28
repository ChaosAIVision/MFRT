"""
Reasoning Path Evaluator for Phase 2 optimization.

This evaluator tracks reasoning paths and classifies them as:
- Good reasoning paths (led to correct answer)
- Bad reasoning paths (led to incorrect answer)
- Alternative reasoning paths (different from ground truth but still correct)

Uses bead-method and production-flow skills for optimization.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from chaos_auto_prompt.interfaces.evaluator import LLMEvaluator
from chaos_auto_prompt.config import get_settings


class ReasoningPathEvaluator(LLMEvaluator):
    """
    Evaluator that tracks and classifies reasoning paths.

    For each model output, it:
    1. Checks if the answer is correct
    2. Extracts the reasoning path from <think> tags
    3. Compares with good paths, bad paths, and ground truth paths
    4. Provides recommendations for prompt updates

    Example usage:
        ```python
        evaluator = ReasoningPathEvaluator(
            feedback_column="reasoning_quality",
            model="gpt-4o",
            good_paths_db=[],  # Will be populated during training
            bad_paths_db=[]    # Will be populated during training
        )

        # Dataset should have columns:
        # - think: model's reasoning in <think> tags
        # - answer: model's final answer
        # - ground_truth: correct answer
        # - ground_truth_reasoning: correct reasoning path

        df, feedback_cols = await evaluator.evaluate(dataset)
        ```
    """

    def __init__(
        self,
        feedback_column: str = "reasoning_quality",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        provider: str = "openai",
        good_paths_db: Optional[List[str]] = None,
        bad_paths_db: Optional[List[str]] = None,
    ):
        """
        Initialize reasoning path evaluator.

        Args:
            feedback_column: Name of the feedback column
            model: LLM model to use for evaluation
            api_key: Optional API key
            provider: LLM provider (openai, google)
            good_paths_db: Database of known good reasoning paths
            bad_paths_db: Database of known bad reasoning paths
        """
        prompt_template = """You are an expert evaluator for reasoning quality in prompt optimization.

MODEL REASONING (from <think> tags):
{reasoning}

MODEL ANSWER:
{answer}

GROUND TRUTH ANSWER:
{ground_truth}

GROUND TRUTH REASONING:
{ground_truth_reasoning}

KNOWN GOOD REASONING PATHS:
{good_paths}

KNOWN BAD REASONING PATHS:
{bad_paths}

Evaluate the model's reasoning:
1. Is the answer correct? (matches ground truth)
2. Does the reasoning path match any known good paths?
3. Does the reasoning path match any known bad paths?
4. If answer is correct but reasoning differs from ground truth, is it a valid alternative approach?

Return a JSON with:
{{
  "reasoning_quality": "<correct|incorrect>",
  "answer_correct": <true|false>,
  "matches_good_path": <true|false>,
  "matches_bad_path": <true|false>,
  "is_alternative_valid": <true|false>,
  "reasoning_type": "<good|bad|alternative|unknown>",
  "explanation": "<detailed analysis>",
  "recommendation": "<should_add_to_good|should_add_to_bad|should_review_good_paths|should_review_bad_paths|no_action>"
}}

Classification logic:
- If answer incorrect AND reasoning matches bad path → reasoning_type: "bad"
- If answer incorrect AND reasoning doesn't match bad path → reasoning_type: "unknown", recommendation: "should_add_to_bad"
- If answer incorrect AND reasoning matches good path → reasoning_type: "bad", recommendation: "should_review_good_paths"
- If answer correct AND reasoning matches good path or ground truth → reasoning_type: "good"
- If answer correct AND reasoning differs from ground truth → reasoning_type: "alternative", consider "should_add_to_good"
"""

        super().__init__(
            feedback_column=feedback_column,
            model=model,
            prompt_template=prompt_template,
            api_key=api_key,
            provider=provider
        )
        self.good_paths_db = good_paths_db or []
        self.bad_paths_db = bad_paths_db or []
        self.settings = get_settings()

        # Track new paths discovered during evaluation
        self.new_good_paths: Set[str] = set()
        self.new_bad_paths: Set[str] = set()

    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        concurrency: int = 20,
        show_progress: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate reasoning paths for all rows.

        Args:
            dataframe: DataFrame with required columns
            concurrency: Number of concurrent API calls
            show_progress: Show progress bar
            **kwargs: Additional parameters

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
        """
        # Validate required columns
        required_columns = ['think', 'answer', 'ground_truth', 'ground_truth_reasoning']
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
            """Evaluate a single row."""
            async with semaphore:
                try:
                    # Extract reasoning from <think> tags
                    reasoning = self._extract_think(row['think'])

                    # Format good and bad paths for prompt
                    good_paths_str = "\n".join([f"- {path}" for path in self.good_paths_db]) or "None"
                    bad_paths_str = "\n".join([f"- {path}" for path in self.bad_paths_db]) or "None"

                    # Format prompt
                    prompt = self.prompt_template.format(
                        reasoning=reasoning,
                        answer=row['answer'],
                        ground_truth=row['ground_truth'],
                        ground_truth_reasoning=row['ground_truth_reasoning'],
                        good_paths=good_paths_str,
                        bad_paths=bad_paths_str,
                        **kwargs
                    )

                    # Generate evaluation
                    messages = [{"role": "user", "content": prompt}]
                    response = await provider.generate_text_with_retry(
                        messages=messages,
                        model=self.model,
                    )

                    # Parse response
                    result = self._parse_response(response)

                    # Track new paths
                    if result.get("recommendation") == "should_add_to_good":
                        self.new_good_paths.add(reasoning)
                    elif result.get("recommendation") == "should_add_to_bad":
                        self.new_bad_paths.add(reasoning)

                    return result

                except Exception as e:
                    print(f"Error evaluating row {idx}: {e}")
                    return {
                        self.feedback_column: "error",
                        "answer_correct": False,
                        "matches_good_path": False,
                        "matches_bad_path": False,
                        "is_alternative_valid": False,
                        "reasoning_type": "unknown",
                        "explanation": f"Error: {str(e)}",
                        "recommendation": "no_action"
                    }

        # Create tasks
        tasks = [evaluate_row(idx, row) for idx, row in dataframe.iterrows()]

        # Execute with progress bar
        if show_progress:
            results = await async_tqdm.gather(
                *tasks,
                desc="Evaluating Reasoning Paths",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )
        else:
            results = await asyncio.gather(*tasks)

        # Add results to dataframe
        df_copy = dataframe.copy()

        feedback_columns = [
            self.feedback_column,
            "answer_correct",
            "matches_good_path",
            "matches_bad_path",
            "is_alternative_valid",
            "reasoning_type",
            "explanation",
            "recommendation"
        ]

        for col_name in feedback_columns:
            df_copy[col_name] = [r.get(col_name) for r in results]

        return df_copy, feedback_columns

    def _extract_think(self, text: str) -> str:
        """
        Extract reasoning content from <think> tags.

        Args:
            text: Text potentially containing <think> tags

        Returns:
            Extracted reasoning content or original text
        """
        if not isinstance(text, str):
            return str(text)

        # Try to extract from <think></think> tags
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return text.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract reasoning evaluation.

        Args:
            response: LLM response text

        Returns:
            Dictionary with evaluation results
        """
        result = {
            self.feedback_column: "unknown",
            "answer_correct": False,
            "matches_good_path": False,
            "matches_bad_path": False,
            "is_alternative_valid": False,
            "reasoning_type": "unknown",
            "explanation": None,
            "recommendation": "no_action"
        }

        try:
            # Try to parse as JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else response

            data = json.loads(json_str)

            result[self.feedback_column] = data.get("reasoning_quality", "unknown")
            result["answer_correct"] = bool(data.get("answer_correct", False))
            result["matches_good_path"] = bool(data.get("matches_good_path", False))
            result["matches_bad_path"] = bool(data.get("matches_bad_path", False))
            result["is_alternative_valid"] = bool(data.get("is_alternative_valid", False))
            result["reasoning_type"] = data.get("reasoning_type", "unknown")
            result["explanation"] = data.get("explanation", "")
            result["recommendation"] = data.get("recommendation", "no_action")

        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            result["explanation"] = f"Parse error: {str(e)}. Response: {response[:200]}"

        return result

    def get_feedback_columns(self) -> List[str]:
        """Get list of feedback columns this evaluator generates."""
        return [
            self.feedback_column,
            "answer_correct",
            "matches_good_path",
            "matches_bad_path",
            "is_alternative_valid",
            "reasoning_type",
            "explanation",
            "recommendation"
        ]

    def get_new_paths(self) -> Dict[str, List[str]]:
        """
        Get newly discovered paths during evaluation.

        Returns:
            Dictionary with 'good_paths' and 'bad_paths' lists
        """
        return {
            "good_paths": list(self.new_good_paths),
            "bad_paths": list(self.new_bad_paths)
        }

    def update_paths_database(self) -> None:
        """Update the paths database with newly discovered paths."""
        self.good_paths_db.extend(self.new_good_paths)
        self.bad_paths_db.extend(self.new_bad_paths)

        # Clear new paths after updating
        self.new_good_paths.clear()
        self.new_bad_paths.clear()


__all__ = ["ReasoningPathEvaluator"]

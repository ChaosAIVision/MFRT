"""
Prompt Learning Optimizer - Core optimization engine.

Refactored from prompt-learning SDK with production-ready architecture.
Uses natural language feedback for prompt optimization via meta-prompt approach.
"""

import asyncio
import copy
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from chaos_auto_prompt.config import get_settings, get_model_context_size
from chaos_auto_prompt.core.dataset_splitter import DatasetSplitter
from chaos_auto_prompt.core.pricing import PricingCalculator
from chaos_auto_prompt.interfaces.token_counter import TiktokenCounter, ApproximateCounter, TokenCounter
from chaos_auto_prompt.interfaces.evaluator import BaseEvaluator
from chaos_auto_prompt.optimizers.meta_prompt import MetaPrompt
from chaos_auto_prompt.providers.base import BaseProvider


# Compile regex pattern once at module level for performance
_TEMPLATE_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class OptimizationError(Exception):
    """Base exception for optimization errors."""
    pass


class DatasetError(Exception):
    """Exception for dataset-related errors."""
    pass


class ProviderError(Exception):
    """Exception for provider-related errors."""
    pass


class PromptLearningOptimizer:
    """
    Prompt Learning Optimizer using meta-prompt approach.

    Optimizes prompts using natural language feedback instead of numerical scores.

    Args:
        prompt: The prompt to optimize (string or list of messages)
        model_choice: Model to use for optimization (default: from settings)
        provider: Optional provider instance for multi-provider support
        token_counter: Optional token counter for batch splitting
        pricing_calculator: Optional pricing calculator for budget tracking
        budget_limit: Budget limit in USD (default: from settings)
        verbose: Enable verbose logging (default: False)

    Example:
        ```python
        from chaos_auto_prompt.optimizers import PromptLearningOptimizer

        optimizer = PromptLearningOptimizer(
            prompt="You are a helpful assistant. Answer: {question}",
            model_choice="gpt-4o",
            budget_limit=5.0
        )

        dataset = pd.DataFrame({
            "question": ["What is the capital of France?"],
            "answer": ["Paris"],
            "feedback": ["correct"]
        })

        optimized = optimizer.optimize(
            dataset=dataset,
            output_column="answer",
            feedback_columns=["feedback"]
        )
        ```
    """

    def __init__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model_choice: Optional[str] = None,
        provider: Optional[BaseProvider] = None,
        token_counter: Optional[TokenCounter] = None,
        pricing_calculator: Optional[PricingCalculator] = None,
        budget_limit: Optional[float] = None,
        verbose: bool = False,
    ):
        self.settings = get_settings()
        self.prompt = prompt
        self.model_choice = model_choice or self.settings.default_model
        self.provider = provider
        self.verbose = verbose
        self.template_variables: List[str] = []

        # Initialize pricing calculator
        if pricing_calculator:
            self.pricing_calculator = pricing_calculator
        else:
            self.pricing_calculator = PricingCalculator(
                budget_limit=budget_limit or self.settings.default_budget
            )

        # Initialize token counter with smart defaults
        if token_counter:
            self.token_counter = token_counter
        elif provider:
            # For non-OpenAI providers, use approximate counting
            self.token_counter = ApproximateCounter()
        else:
            # For OpenAI, use tiktoken
            self.token_counter = TiktokenCounter()

        # Initialize meta-prompter
        self.meta_prompter = MetaPrompt()

    def _load_dataset(self, dataset: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """Load dataset from DataFrame or JSON file."""
        if isinstance(dataset, pd.DataFrame):
            return dataset
        elif isinstance(dataset, str):
            try:
                return pd.read_json(dataset)
            except Exception as e:
                raise DatasetError(f"Failed to load dataset from {dataset}: {e}")
        else:
            raise DatasetError(f"Invalid dataset type: {type(dataset)}")

    def _validate_inputs(
        self,
        dataset: pd.DataFrame,
        feedback_columns: List[str],
        output_column: Optional[str] = None,
        output_required: bool = False,
    ):
        """Validate that we have the necessary inputs for optimization."""
        if not feedback_columns:
            raise DatasetError(
                "feedback_columns must be provided. "
                "Need feedback for meta-prompt optimization."
            )

        required_columns = []
        if output_required:
            if output_column is None:
                raise DatasetError("output_column must be provided")
            required_columns.append(output_column)
        required_columns.extend(feedback_columns)

        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            raise DatasetError(f"Dataset missing required columns: {missing_columns}")

    def _extract_prompt_content(self) -> str:
        """Extract prompt content from various formats."""
        if isinstance(self.prompt, str):
            return self.prompt
        elif isinstance(self.prompt, list):
            for message in self.prompt:
                if message.get("role") == "system":
                    return message.get("content", "")
            # If no system message, return first user message
            for message in self.prompt:
                if message.get("role") == "user":
                    return message.get("content", "")
            raise ValueError("No system or user message found in the prompt")
        else:
            raise ValueError("Prompt must be a string or list of messages")

    def _detect_template_variables(self, prompt_content: str) -> List[str]:
        """Return unique {placeholders} that look like template vars."""
        return list({m.group(1) for m in _TEMPLATE_RE.finditer(prompt_content)})

    def _create_optimized_prompt(
        self, optimized_content: str
    ) -> Union[str, List[Dict[str, str]]]:
        """Create optimized prompt in the same format as input."""
        if isinstance(self.prompt, str):
            return optimized_content
        elif isinstance(self.prompt, list):
            optimized_messages = copy.deepcopy(self.prompt)
            for i, message in enumerate(optimized_messages):
                if message.get("role") in ("system", "user"):
                    optimized_messages[i]["content"] = optimized_content
                    break
            return optimized_messages
        else:
            raise ValueError("Invalid prompt format")

    async def _generate_with_provider(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """Generate text using the configured provider."""
        if self.provider:
            return await self.provider.generate_text_with_retry(
                messages=messages,
                model=self.model_choice,
            )
        else:
            # Default to OpenAI if no provider specified
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            response = await client.chat.completions.create(
                model=self.model_choice,
                messages=messages,
            )
            return response.choices[0].message.content or ""

    async def run_evaluators(
        self,
        dataset: pd.DataFrame,
        evaluators: List[BaseEvaluator],
        feedback_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run evaluators on dataset to generate feedback columns automatically.

        This method applies evaluators to the dataset, generating feedback columns
        like "correctness", "explanation", etc. without requiring manual feedback.

        Args:
            dataset: DataFrame with model outputs to evaluate
            evaluators: List of evaluator instances to run
            feedback_columns: Optional list of expected feedback column names
                             (used for validation)

        Returns:
            Tuple of (updated_dataset, generated_feedback_columns)
            - updated_dataset: Dataset with new feedback columns added
            - generated_feedback_columns: List of column names that were added

        Example:
            ```python
            from chaos_auto_prompt.evaluators import ClassificationEvaluator

            evaluator = ClassificationEvaluator(
                feedback_column="correctness",
                model="gpt-4o",
                prompt_template="Evaluate: {output}",
                choices={"correct": 1, "incorrect": 0}
            )

            dataset, feedback_cols = optimizer.run_evaluators(
                dataset,
                evaluators=[evaluator],
                feedback_columns=["correctness", "explanation"]
            )
            ```
        """
        updated_dataset = dataset.copy()
        all_feedback_columns = []

        if self.verbose:
            print(f"Running {len(evaluators)} evaluator(s) on {len(dataset)} rows")

        for evaluator in evaluators:
            try:
                # Run evaluator
                updated_dataset, feedback_cols = await evaluator.evaluate(updated_dataset)
                all_feedback_columns.extend(feedback_cols)

                if self.verbose:
                    print(f"  ✓ {evaluator.__class__.__name__}: generated {feedback_cols}")

            except Exception as e:
                print(f"  ✗ {evaluator.__class__.__name__} failed: {e}")
                # Initialize columns with None if evaluator failed
                for col in evaluator.get_feedback_columns():
                    if col not in updated_dataset.columns:
                        updated_dataset[col] = None

        # Validate if feedback_columns was provided
        if feedback_columns:
            missing = set(feedback_columns) - set(all_feedback_columns)
            if missing:
                print(f"Warning: Expected feedback columns not generated: {missing}")

        return updated_dataset, list(set(all_feedback_columns))

    async def optimize(
        self,
        dataset: Union[pd.DataFrame, str],
        output_column: str,
        feedback_columns: List[str],
        context_size: Optional[int] = None,
        ruleset: Optional[str] = None,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Optimize the prompt using meta-prompt approach.

        Args:
            dataset: DataFrame or path to JSON file
            output_column: Name of column with LLM outputs
            feedback_columns: List of column names with feedback
            context_size: Context window size in tokens (default: from model)
            ruleset: Optional ruleset for coding agent optimization

        Returns:
            Optimized prompt in same format as input
        """
        # Load and validate dataset
        dataset = self._load_dataset(dataset)
        self._validate_inputs(dataset, feedback_columns, output_column, output_required=True)

        # Extract prompt content and detect template variables
        prompt_content = self._extract_prompt_content()
        self.template_variables = self._detect_template_variables(prompt_content)

        # Initialize dataset splitter
        splitter = DatasetSplitter(self.token_counter)
        context_size = context_size or get_model_context_size(self.model_choice)

        # Create batches
        columns_to_count = list(dataset.columns)
        batch_dataframes = splitter.split_into_batches(
            dataset, columns_to_count, context_size
        )

        if self.verbose:
            print(f"Processing {len(dataset)} examples in {len(batch_dataframes)} batches")

        # Process batches
        optimized_prompt_content = prompt_content

        for i, batch in enumerate(batch_dataframes):
            try:
                # Construct meta-prompt
                meta_prompt_content = self.meta_prompter.construct_content(
                    batch_df=batch,
                    prompt_to_optimize_content=optimized_prompt_content,
                    template_variables=self.template_variables,
                    feedback_columns=feedback_columns,
                    output_column=output_column,
                    ruleset=ruleset,
                )

                # Check budget
                input_tokens = len(meta_prompt_content) // 4
                output_tokens = 1000

                if self.pricing_calculator.would_exceed_budget(
                    self.model_choice, input_tokens, output_tokens
                ):
                    print(
                        f"Budget limit ${self.pricing_calculator.budget_limit:.2f} exceeded. "
                        f"Current cost: ${self.pricing_calculator.get_total_cost():.4f}"
                    )
                    break

                # Generate optimized prompt
                messages = [{"role": "user", "content": meta_prompt_content}]
                response_text = await self._generate_with_provider(messages)

                # Track costs
                output_tokens = len(response_text) // 4
                cost = self.pricing_calculator.add_usage(
                    self.model_choice, input_tokens, output_tokens
                )

                if self.verbose:
                    print(
                        f"Batch {i + 1}/{len(batch_dataframes)}: "
                        f"Cost ${cost:.4f} (Total: ${self.pricing_calculator.get_total_cost():.4f})"
                    )

                # Update optimized prompt
                if ruleset:
                    ruleset = response_text
                else:
                    optimized_prompt_content = response_text

            except (ProviderError, OptimizationError) as e:
                print(f"Batch {i + 1}/{len(batch_dataframes)}: Failed - {e}")
                continue

        if ruleset:
            return ruleset

        return self._create_optimized_prompt(optimized_prompt_content)


# Export convenience
__all__ = [
    "PromptLearningOptimizer",
    "OptimizationError",
    "DatasetError",
    "ProviderError",
]

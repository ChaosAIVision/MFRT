"""
Intent-Aware Prompt Optimizer - 3-Phase optimization with user intent extraction.

This optimizer extends the base PromptLearningOptimizer with intent-aware optimization:
- Phase 0: Extract user's intent definition from system prompt + examples
- Phase 1: Construction grounded in user's intent
- Phase 2: Reasoning that validates against user's intent

The intent definition includes:
- Concept name (e.g., "milestone", "spam", "urgent")
- User's specific definition
- Positive/negative indicators
- Boundary cases

This ensures the optimized prompt aligns with the user's specific understanding
of the classification concept, not a generic definition.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any

import pandas as pd

from ..config import get_model_context_size
from ..core.dataset_splitter import DatasetSplitter
from ..evaluators.two_stage import TwoStageEvaluator
from ..interfaces.evaluator import BaseEvaluator
from ..utils.intent_extractor import IntentExtractor
from .prompt_optimizer import PromptLearningOptimizer, OptimizationError, DatasetError

logger = logging.getLogger(__name__)


class IntentAwarePromptOptimizer(PromptLearningOptimizer):
    """
    Intent-aware prompt optimizer with 3-phase optimization flow.

    Extends PromptLearningOptimizer with intent extraction and grounding.

    Args:
        prompt: The prompt to optimize (string or list of messages)
        model_choice: Model to use for optimization (default: from settings)
        provider: Optional provider instance for multi-provider support
        token_counter: Optional token counter for batch splitting
        pricing_calculator: Optional pricing calculator for budget tracking
        budget_limit: Budget limit in USD (default: from settings)
        verbose: Enable verbose logging (default: False)
        extract_intent: Whether to extract intent from prompt+examples (default: True)
        intent: Optional pre-extracted intent (skips Phase 0 if provided)

    Example:
        ```python
        from chaos_auto_prompt.optimizers import IntentAwarePromptOptimizer

        optimizer = IntentAwarePromptOptimizer(
            prompt="classify this article contain milestone or not",
            model_choice="gpt-4o",
            budget_limit=5.0
        )

        dataset = pd.DataFrame({
            "input": [
                "Player scores 58th goal, breaking team record",
                "Coach announces retirement",
            ],
            "output": [True, False]
        })

        optimized = await optimizer.optimize(
            dataset=dataset,
            output_column="output",
            feedback_columns=["two_stage_feedback"]
        )
        ```
    """

    def __init__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model_choice: Optional[str] = None,
        provider: Optional["BaseProvider"] = None,
        token_counter: Optional["TokenCounter"] = None,
        pricing_calculator: Optional["PricingCalculator"] = None,
        budget_limit: Optional[float] = None,
        verbose: bool = False,
        extract_intent: bool = True,
        intent: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            prompt=prompt,
            model_choice=model_choice,
            provider=provider,
            token_counter=token_counter,
            pricing_calculator=pricing_calculator,
            budget_limit=budget_limit,
            verbose=verbose,
        )

        self.extract_intent = extract_intent
        self.intent = intent
        self.intent_extractor: Optional[IntentExtractor] = None

        # Initialize IntentExtractor if extraction is enabled
        if self.extract_intent and not self.intent:
            if not provider:
                raise ValueError(
                    "Provider required for intent extraction. "
                    "Either pass provider or set extract_intent=False"
                )
            self.intent_extractor = IntentExtractor(
                provider=provider,
                model=model_choice
            )

    async def optimize(
        self,
        dataset: Union[pd.DataFrame, str],
        output_column: str,
        feedback_columns: Optional[List[str]] = None,
        context_size: Optional[int] = None,
        ruleset: Optional[str] = None,
        input_column: str = "input",
        use_two_stage_evaluator: bool = True,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Optimize prompt with 3-phase intent-aware flow.

        Phase 0: Intent Extraction (if enabled and not provided)
        Phase 1: Intent-grounded construction
        Phase 2: Intent-aware reasoning

        Args:
            dataset: DataFrame or path to JSON file
            output_column: Name of column with LLM outputs or ground truth labels
            feedback_columns: List of feedback columns (optional, auto-generated if None)
            context_size: Context window size in tokens (default: from model)
            ruleset: Optional ruleset for coding agent optimization
            input_column: Name of column with input text (for intent extraction)
            use_two_stage_evaluator: Whether to use TwoStageEvaluator (default: True)

        Returns:
            Optimized prompt in same format as input

        Raises:
            DatasetError: If dataset validation fails
            OptimizationError: If optimization fails
        """
        # Load and validate dataset
        dataset = self._load_dataset(dataset)

        # Phase 0: Extract Intent (if enabled and not already provided)
        if self.extract_intent and not self.intent:
            if self.verbose:
                print("\n=== Phase 0: Intent Extraction ===")

            prompt_content = self._extract_prompt_content()

            try:
                self.intent = await self.intent_extractor.extract_intent(
                    system_prompt=prompt_content,
                    examples=dataset,
                    output_column=output_column,
                    input_column=input_column,
                    max_examples=10,
                )

                if self.verbose:
                    concept = self.intent.get("concept", "unknown")
                    confidence = self.intent.get("confidence", 0.0)
                    print(f"✓ Intent extracted: '{concept}' (confidence: {confidence:.2f})")
                    print(f"  Definition: {self.intent.get('definition', '')[:100]}...")
                    print(f"  Positive indicators: {len(self.intent.get('positive_indicators', []))}")
                    print(f"  Negative indicators: {len(self.intent.get('negative_indicators', []))}")

            except Exception as e:
                logger.error(f"Intent extraction failed: {e}")
                if self.verbose:
                    print(f"⚠️  Intent extraction failed: {e}")
                    print("  Continuing without intent grounding...")
                self.intent = None

        # Generate feedback using TwoStageEvaluator if requested
        if use_two_stage_evaluator:
            if feedback_columns is None:
                feedback_columns = []

            if self.verbose:
                print("\n=== Generating Feedback with TwoStageEvaluator ===")

            # Create TwoStageEvaluator with intent
            evaluator = TwoStageEvaluator(
                feedback_column="two_stage_feedback",
                construction_weight=0.5,
                reasoning_weight=0.5,
                groundtruth_column=output_column,
                prediction_column=output_column,  # Using output as prediction for training data
                intent=self.intent,
            )

            # Run evaluator
            dataset, generated_feedback_columns = await self.run_evaluators(
                dataset=dataset,
                evaluators=[evaluator],
                feedback_columns=feedback_columns,
            )

            # Use generated feedback columns
            feedback_columns = generated_feedback_columns

            if self.verbose:
                print(f"✓ Generated {len(feedback_columns)} feedback columns: {feedback_columns}")

        # Validate after evaluation
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
            print(f"\n=== Phase 1-2: Intent-Grounded Optimization ===")
            print(f"Processing {len(dataset)} examples in {len(batch_dataframes)} batches")

        # Process batches
        optimized_prompt_content = prompt_content

        for i, batch in enumerate(batch_dataframes):
            try:
                # Construct meta-prompt (intent-aware if intent available)
                if self.intent:
                    meta_prompt_content = self.meta_prompter.construct_intent_aware_content(
                        batch_df=batch,
                        prompt_to_optimize_content=optimized_prompt_content,
                        template_variables=self.template_variables,
                        feedback_columns=feedback_columns,
                        output_column=output_column,
                        intent=self.intent,
                        annotations=None,
                    )
                else:
                    # Fallback to regular meta-prompt if no intent
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

                    # Auto-fill any remaining placeholders (safety net)
                    import re
                    placeholders_before = re.findall(r'\{examples?(\d+)\}', optimized_prompt_content)
                    if placeholders_before:
                        optimized_prompt_content = self._auto_fill_example_placeholders(
                            optimized_prompt_content,
                            dataset
                        )
                        if self.verbose:
                            print(
                                f"  ⚠️  Auto-filled {len(set(placeholders_before))} placeholder(s): "
                                f"{set(placeholders_before)}"
                            )

            except Exception as e:
                logger.error(f"Batch {i + 1} optimization failed: {e}")
                print(f"Batch {i + 1}/{len(batch_dataframes)}: Failed - {e}")
                continue

        if ruleset:
            return ruleset

        if self.verbose:
            print("\n=== Optimization Complete ===")
            if self.intent:
                print(f"✓ Intent-aware optimization completed")
                print(f"  Concept: {self.intent.get('concept', 'unknown')}")
            else:
                print(f"✓ Standard optimization completed (no intent)")

        return self._create_optimized_prompt(optimized_prompt_content)

    def get_intent(self) -> Optional[Dict[str, Any]]:
        """
        Get the extracted intent definition.

        Returns:
            Intent dictionary or None if not extracted
        """
        return self.intent


__all__ = ["IntentAwarePromptOptimizer"]

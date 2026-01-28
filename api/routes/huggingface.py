"""
HuggingFace dataset optimization endpoint.

This module provides API endpoints for optimizing prompts using datasets
from HuggingFace Hub with automatic evaluation and feedback generation.
"""

import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from fastapi import APIRouter, HTTPException, status

from chaos_auto_prompt.config import get_settings
from chaos_auto_prompt.evaluators.classification import ClassificationEvaluator
from chaos_auto_prompt.optimizers import PromptLearningOptimizer
from chaos_auto_prompt.providers import OpenAIProvider, GoogleProvider, ProviderConfig

from ..models import (
    HuggingFaceOptimizeRequest,
    HuggingFaceOptimizeResponse,
    UsageSummary,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimize", tags=["optimization"])


@router.post(
    "/huggingface",
    response_model=HuggingFaceOptimizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Optimize prompt from HuggingFace dataset",
    description="""
    Load a dataset from HuggingFace Hub and optimize a prompt based on
    input-output pairs with optional automatic feedback generation.

    **Workflow:**
    1. Load dataset from HuggingFace Hub
    2. Extract system_prompt, input, output columns
    3. Optionally run evaluators to generate feedback
    4. Optimize prompt using feedback
    5. Return optimized prompt + metrics

    **Example Request:**
    ```json
    {
        "dataset_name": "user/abbott-chatbot-dataset",
        "system_prompt_column": "system_prompt",
        "input_column": "customer_question",
        "output_column": "chatbot_answer",
        "evaluators": [{
            "type": "classification",
            "feedback_column": "quality",
            "model": "gpt-4o",
            "prompt_template": "Evaluate this chatbot response...",
            "choices": {"excellent": 2, "good": 1, "poor": 0}
        }],
        "max_samples": 100,
        "budget": 5.0
    }
    ```
    """,
)
async def optimize_from_huggingface(
    request: HuggingFaceOptimizeRequest,
) -> HuggingFaceOptimizeResponse:
    """
    Optimize a prompt using a HuggingFace dataset.

    Args:
        request: HuggingFace optimization request with dataset details

    Returns:
        HuggingFaceOptimizeResponse with optimized prompt and metrics

    Raises:
        HTTPException: On dataset loading errors or optimization failures
    """
    settings = get_settings()

    try:
        # Step 1: Load dataset from HuggingFace
        logger.info(
            f"Loading dataset {request.dataset_name} "
            f"(config={request.dataset_config}, split={request.dataset_split})"
        )

        try:
            if request.dataset_config:
                hf_dataset = load_dataset(
                    request.dataset_name,
                    request.dataset_config,
                    split=request.dataset_split,
                )
            else:
                hf_dataset = load_dataset(
                    request.dataset_name,
                    split=request.dataset_split,
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load HuggingFace dataset: {str(e)}",
            )

        # Convert to pandas DataFrame
        df = pd.DataFrame(hf_dataset)

        # Limit samples if requested
        if request.max_samples and len(df) > request.max_samples:
            df = df.head(request.max_samples)
            logger.info(f"Limited dataset to {request.max_samples} samples")

        # Validate required columns exist
        required_columns = [
            request.system_prompt_column,
            request.input_column,
            request.output_column,
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}",
            )

        # Step 1.5: Train/Test split
        from sklearn.model_selection import train_test_split

        train_size = request.train_split or 0.7  # Default 70% train, 30% test

        if train_size < 1.0:  # Only split if train_size < 1.0
            try:
                stratify_column = df[request.output_column] if request.stratify else None
                train_df, test_df = train_test_split(
                    df,
                    train_size=train_size,
                    random_state=42,
                    stratify=stratify_column
                )
                logger.info(f"Split dataset: {len(train_df)} train, {len(test_df)} test")
            except Exception as e:
                logger.warning(f"Failed to stratify split: {e}. Using random split.")
                train_df, test_df = train_test_split(
                    df,
                    train_size=train_size,
                    random_state=42
                )
                logger.info(f"Split dataset (no stratify): {len(train_df)} train, {len(test_df)} test")
        else:
            # No split - use full dataset as both train and test
            train_df = df.copy()
            test_df = df.copy()
            logger.info(f"No split - using full dataset ({len(df)} samples)")

        # Step 2: Extract initial prompt from first row
        initial_prompt = df[request.system_prompt_column].iloc[0]
        logger.info(f"Initial prompt: {initial_prompt[:100]}...")

        # Step 3: Set model and budget
        model = request.model or settings.openai_default_model
        budget = request.budget or settings.default_budget

        # Step 3.5: Generate predictions with INITIAL prompt on TRAIN set
        logger.info("Generating predictions on train set with initial prompt")

        from chaos_auto_prompt.providers import OpenAIProvider, ProviderConfig

        provider_config = ProviderConfig(
            api_key=settings.openai_api_key,
            model=model,
        )
        provider = OpenAIProvider(config=provider_config)

        # Generate predictions for each row in TRAIN set
        predictions = []
        for idx, row in train_df.iterrows():
            try:
                input_value = row[request.input_column]

                # Create messages with initial system prompt
                messages = [
                    {"role": "system", "content": initial_prompt},
                    {"role": "user", "content": str(input_value)}
                ]

                # Generate prediction
                prediction = await provider.generate_text_with_retry(
                    messages=messages,
                    model=model,
                )

                predictions.append(prediction)

            except Exception as e:
                logger.warning(f"Failed to generate prediction for row {idx}: {e}")
                # Fallback to original output if generation fails
                predictions.append(row[request.output_column])

        # Add predictions to train dataframe
        train_df = train_df.copy()
        train_df["predicted_output"] = predictions

        logger.info(f"Generated {len(predictions)} predictions on train set")

        # Step 4: Run evaluators if provided
        feedback_columns = request.feedback_columns.copy()

        if request.evaluators:
            logger.info(f"Running {len(request.evaluators)} evaluator(s)")

            evaluator_instances = []
            for eval_config in request.evaluators:
                if eval_config.type == "classification":
                    evaluator = ClassificationEvaluator(
                        feedback_column=eval_config.feedback_column,
                        model=eval_config.model,
                        prompt_template=eval_config.prompt_template,
                        choices=eval_config.choices or {"correct": 1, "incorrect": 0},
                        include_explanation=eval_config.include_explanation,
                    )
                    evaluator_instances.append(evaluator)
                else:
                    logger.warning(f"Unknown evaluator type: {eval_config.type}")

            # Run evaluators on TRAIN set (with initial predictions)
            temp_optimizer = PromptLearningOptimizer(
                prompt=initial_prompt,
                model_choice=model,
                budget_limit=budget,
                verbose=request.verbose,
            )

            train_df, generated_feedback_cols = await temp_optimizer.run_evaluators(
                dataset=train_df,
                evaluators=evaluator_instances,
            )

            feedback_columns.extend(generated_feedback_cols)
            logger.info(f"Generated feedback columns: {generated_feedback_cols}")

        # Validate feedback columns exist
        if not feedback_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No feedback columns provided. Either provide feedback_columns "
                "or evaluators to generate them.",
            )

        missing_feedback = [col for col in feedback_columns if col not in train_df.columns]
        if missing_feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing feedback columns: {missing_feedback}. "
                f"Available columns: {list(train_df.columns)}",
            )

        # Step 5: Initialize loop variables
        max_loops = request.max_loops or 1
        threshold = request.threshold or 0.95

        current_prompt = initial_prompt
        prompts_history = [initial_prompt]
        train_metrics_history = []
        test_metrics_history = []

        loop_iteration = 0
        target_reached = False
        last_optimizer = None  # Track last optimizer for pricing info

        logger.info(f"Starting optimization loop: max_loops={max_loops}, threshold={threshold}")

        # Helper function to calculate metrics
        def calculate_metrics(dataframe: pd.DataFrame, columns: list) -> dict:
            """Helper to calculate metrics from feedback columns."""
            metrics = {}
            for feedback_col in columns:
                if feedback_col not in dataframe.columns:
                    continue

                if pd.api.types.is_numeric_dtype(dataframe[feedback_col]):
                    metrics[feedback_col] = {
                        "mean": float(dataframe[feedback_col].mean()),
                        "std": float(dataframe[feedback_col].std()),
                        "min": float(dataframe[feedback_col].min()),
                        "max": float(dataframe[feedback_col].max()),
                    }
                else:
                    # For categorical feedback
                    value_counts = dataframe[feedback_col].value_counts().to_dict()
                    metrics[feedback_col] = {
                        "distribution": {str(k): int(v) for k, v in value_counts.items()}
                    }
            return metrics

        # Calculate baseline metrics on TRAIN set (before any optimization)
        metrics_before = calculate_metrics(train_df, feedback_columns)

        # Create provider for generating predictions
        from chaos_auto_prompt.providers import OpenAIProvider, ProviderConfig

        provider_config = ProviderConfig(
            api_key=settings.openai_api_key,
            model=model,
        )
        provider = OpenAIProvider(config=provider_config)

        # Step 5.5: Initial test evaluation with initial_prompt
        logger.info("Evaluating initial prompt on test set")

        initial_test_outputs = []
        for idx, row in test_df.iterrows():
            try:
                input_value = row[request.input_column]
                messages = [
                    {"role": "system", "content": initial_prompt},
                    {"role": "user", "content": str(input_value)}
                ]
                response = await provider.generate_text_with_retry(
                    messages=messages,
                    model=model,
                )
                initial_test_outputs.append(response)
            except Exception as e:
                logger.warning(f"Failed to generate initial test output for row {idx}: {e}")
                initial_test_outputs.append(row[request.output_column])

        # Create test dataframe with initial predictions
        test_df_initial = test_df.copy()
        test_df_initial["predicted_output"] = initial_test_outputs

        # Run evaluators on initial test predictions
        if request.evaluators:
            temp_optimizer_test = PromptLearningOptimizer(
                prompt=initial_prompt,
                model_choice=model,
                budget_limit=budget,
                verbose=request.verbose,
            )
            test_df_initial, _ = await temp_optimizer_test.run_evaluators(
                dataset=test_df_initial,
                evaluators=evaluator_instances,
            )

        # Calculate initial test metrics
        initial_test_metrics = calculate_metrics(test_df_initial, feedback_columns)
        logger.info(f"Initial test metrics: {initial_test_metrics}")

        # Check if initial prompt already meets threshold
        if feedback_columns:
            first_feedback = feedback_columns[0]
            if first_feedback in initial_test_metrics:
                metric = initial_test_metrics[first_feedback]

                # For numeric metrics
                if "mean" in metric:
                    accuracy = metric["mean"]
                    logger.info(f"Initial test accuracy: {accuracy:.4f}, threshold: {threshold}")
                    if accuracy >= threshold:
                        logger.info(f"Initial prompt already meets threshold! Skipping optimization.")
                        target_reached = True

                # For categorical metrics
                elif "distribution" in metric:
                    dist = metric["distribution"]
                    positive_labels = ["correct", "good", "excellent", "yes", "true"]
                    positive_count = sum(dist.get(label, 0) for label in positive_labels)
                    total = sum(dist.values())
                    accuracy = (positive_count / total) if total > 0 else 0
                    logger.info(f"Initial test accuracy: {accuracy:.4f}, threshold: {threshold}")
                    if accuracy >= threshold:
                        logger.info(f"Initial prompt already meets threshold! Skipping optimization.")
                        target_reached = True

        # If target already reached, skip loop and return early
        if target_reached:
            logger.info("Returning early - initial prompt is already good enough")
            metrics_after = initial_test_metrics
            optimized_prompt = initial_prompt

        # Step 6: Optimization loop
        while loop_iteration < max_loops and not target_reached:
            loop_iteration += 1
            logger.info(f"=== Loop iteration {loop_iteration}/{max_loops} ===")

            # Step 6.1: Optimize prompt on TRAIN set
            logger.info(f"Optimizing prompt on train set (iteration {loop_iteration})")

            optimizer = PromptLearningOptimizer(
                prompt=current_prompt,
                model_choice=model,
                budget_limit=budget,
                verbose=request.verbose,
            )

            optimized_prompt = await optimizer.optimize(
                dataset=train_df,
                output_column="predicted_output",  # Use model predictions, not ground truth
                feedback_columns=feedback_columns,
            )

            # Save optimizer for pricing info
            last_optimizer = optimizer

            # Update current prompt
            current_prompt = optimized_prompt
            prompts_history.append(optimized_prompt)

            logger.info(f"Optimized prompt (iteration {loop_iteration}): {optimized_prompt[:100]}...")

            # Step 6.2: Re-evaluate on TRAIN set with optimized prompt
            logger.info(f"Re-generating predictions on train set with optimized prompt (iteration {loop_iteration})")

            train_outputs = []
            for idx, row in train_df.iterrows():
                try:
                    input_value = row[request.input_column]
                    messages = [
                        {"role": "system", "content": optimized_prompt},
                        {"role": "user", "content": str(input_value)}
                    ]
                    response = await provider.generate_text_with_retry(
                        messages=messages,
                        model=model,
                    )
                    train_outputs.append(response)
                except Exception as e:
                    logger.warning(f"Failed to generate train output for row {idx}: {e}")
                    train_outputs.append(row[request.output_column])

            # Update train_df with new predictions
            train_df_eval = train_df.copy()
            train_df_eval["predicted_output"] = train_outputs

            # Re-run evaluators on train set
            if request.evaluators:
                train_df_eval, _ = await optimizer.run_evaluators(
                    dataset=train_df_eval,
                    evaluators=evaluator_instances,
                )

            # Calculate train metrics
            train_metrics = calculate_metrics(train_df_eval, feedback_columns)
            train_metrics_history.append(train_metrics)

            logger.info(f"Train metrics (iteration {loop_iteration}): {train_metrics}")

            # Update train_df for next iteration
            train_df = train_df_eval.copy()

            # Step 6.3: Evaluate on TEST set with optimized prompt
            logger.info(f"Evaluating on test set with optimized prompt (iteration {loop_iteration})")

            test_outputs = []
            for idx, row in test_df.iterrows():
                try:
                    input_value = row[request.input_column]
                    messages = [
                        {"role": "system", "content": optimized_prompt},
                        {"role": "user", "content": str(input_value)}
                    ]
                    response = await provider.generate_text_with_retry(
                        messages=messages,
                        model=model,
                    )
                    test_outputs.append(response)
                except Exception as e:
                    logger.warning(f"Failed to generate test output for row {idx}: {e}")
                    test_outputs.append(row[request.output_column])

            # Create test dataframe with predictions
            test_df_eval = test_df.copy()
            test_df_eval["predicted_output"] = test_outputs

            # Re-run evaluators on test set
            if request.evaluators:
                test_df_eval, _ = await optimizer.run_evaluators(
                    dataset=test_df_eval,
                    evaluators=evaluator_instances,
                )

            # Calculate test metrics
            test_metrics = calculate_metrics(test_df_eval, feedback_columns)
            test_metrics_history.append(test_metrics)

            logger.info(f"Test metrics (iteration {loop_iteration}): {test_metrics}")

            # Step 6.4: Check threshold on first feedback column
            if feedback_columns:
                first_feedback = feedback_columns[0]
                if first_feedback in test_metrics:
                    metric = test_metrics[first_feedback]

                    # For numeric metrics, check mean
                    if "mean" in metric:
                        accuracy = metric["mean"]
                        logger.info(f"Test accuracy: {accuracy:.4f}, threshold: {threshold}")
                        if accuracy >= threshold:
                            target_reached = True
                            logger.info(f"Target accuracy reached! Stopping at iteration {loop_iteration}")

                    # For categorical metrics, check positive percentage
                    elif "distribution" in metric:
                        dist = metric["distribution"]
                        positive_labels = ["correct", "good", "excellent", "yes", "true"]
                        positive_count = sum(dist.get(label, 0) for label in positive_labels)
                        total = sum(dist.values())
                        accuracy = (positive_count / total) if total > 0 else 0
                        logger.info(f"Test accuracy: {accuracy:.4f}, threshold: {threshold}")
                        if accuracy >= threshold:
                            target_reached = True
                            logger.info(f"Target accuracy reached! Stopping at iteration {loop_iteration}")

        # Final metrics after loop
        metrics_after = test_metrics_history[-1] if test_metrics_history else calculate_metrics(test_df, feedback_columns)
        optimized_prompt = prompts_history[-1]

        logger.info(f"Optimization loop completed: {loop_iteration} iterations, target_reached={target_reached}")

        # Determine why optimization stopped
        if loop_iteration == 0 and target_reached:
            stopped_reason = "early_return"  # Initial prompt already met threshold
        elif target_reached:
            stopped_reason = "threshold_reached"  # Threshold met during optimization
        elif loop_iteration >= max_loops:
            stopped_reason = "max_loops"  # Reached max iterations
        else:
            stopped_reason = "early_return"  # Other early return (shouldn't happen normally)

        logger.info(f"Stopped reason: {stopped_reason}")

        # Step 7: Calculate improvement
        improvement = {}

        for feedback_col in feedback_columns:
            if feedback_col not in metrics_before or feedback_col not in metrics_after:
                continue

            before = metrics_before[feedback_col]
            after = metrics_after[feedback_col]

            if "mean" in before and "mean" in after:
                # Numeric metrics
                improvement[feedback_col] = {
                    "before_mean": before["mean"],
                    "after_mean": after["mean"],
                    "absolute_change": after["mean"] - before["mean"],
                    "percent_change": ((after["mean"] - before["mean"]) / before["mean"] * 100) if before["mean"] != 0 else 0,
                }
            elif "distribution" in before and "distribution" in after:
                # Categorical metrics - calculate improvement in "good" labels
                before_dist = before["distribution"]
                after_dist = after["distribution"]

                # Try to find positive labels (good, excellent, correct, etc.)
                positive_labels = ["good", "excellent", "correct", "yes", "true"]
                positive_count_before = sum(before_dist.get(label, 0) for label in positive_labels)
                positive_count_after = sum(after_dist.get(label, 0) for label in positive_labels)

                total = sum(before_dist.values())

                improvement[feedback_col] = {
                    "before_distribution": before_dist,
                    "after_distribution": after_dist,
                    "positive_count_before": positive_count_before,
                    "positive_count_after": positive_count_after,
                    "improvement": positive_count_after - positive_count_before,
                    "before_positive_pct": (positive_count_before / total * 100) if total > 0 else 0,
                    "after_positive_pct": (positive_count_after / total * 100) if total > 0 else 0,
                }

        # Step 11: Build response
        return HuggingFaceOptimizeResponse(
            success=True,
            initial_prompt=initial_prompt,
            optimized_prompt=optimized_prompt,
            dataset_info={
                "name": request.dataset_name,
                "config": request.dataset_config,
                "split": request.dataset_split,
                "num_samples": len(df),
                "columns": list(df.columns),
                "system_prompt_column": request.system_prompt_column,
                "input_column": request.input_column,
                "output_column": request.output_column,
            },
            usage_summary=UsageSummary(
                total_input_tokens=last_optimizer.pricing_calculator.total_input_tokens if last_optimizer else 0,
                total_output_tokens=last_optimizer.pricing_calculator.total_output_tokens if last_optimizer else 0,
                total_tokens=last_optimizer.pricing_calculator.total_tokens if last_optimizer else 0,
                total_cost=last_optimizer.pricing_calculator.get_total_cost() if last_optimizer else 0,
                budget_limit=budget,
                remaining_budget=budget - (last_optimizer.pricing_calculator.get_total_cost() if last_optimizer else 0),
                budget_usage_percentage=((last_optimizer.pricing_calculator.get_total_cost() / budget * 100) if last_optimizer else 0) if budget else 0,
            ),
            iterations=[],  # TODO: Capture iteration history from optimizer
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
            train_metrics=train_metrics_history,
            test_metrics=test_metrics_history,
            prompts_history=prompts_history,
            num_loops=loop_iteration,
            stopped_reason=stopped_reason,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in HuggingFace optimization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )

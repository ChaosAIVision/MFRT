"""
Optimization API endpoints.

Provides REST API for prompt optimization using various providers.
"""

import pandas as pd
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.models import (
    OptimizeRequest,
    OptimizeResponse,
    ErrorResponse,
    UsageSummary,
    Message,
    EvaluatorConfig,
)
from chaos_auto_prompt.optimizers import PromptLearningOptimizer, ProviderError, OptimizationError
from chaos_auto_prompt.evaluators import ClassificationEvaluator
from chaos_auto_prompt.config import get_settings
from chaos_auto_prompt.providers.openai import OpenAIProvider
from chaos_auto_prompt.providers.google import GoogleProvider
from chaos_auto_prompt.providers.base import ProviderConfig

router = APIRouter(prefix="/api/v1", tags=["optimization"])
settings = get_settings()


def calculate_metrics(dataframe: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    Helper to calculate metrics from feedback columns.

    Supports both numeric (mean/std/min/max) and categorical (distribution) metrics.
    """
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


def calculate_improvement(
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
    feedback_columns: List[str]
) -> Dict[str, Any]:
    """Calculate improvement between before and after metrics."""
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
            # Categorical metrics
            before_dist = before["distribution"]
            after_dist = after["distribution"]

            # Find positive labels (good, excellent, correct, etc.)
            positive_labels = ["good", "excellent", "correct", "yes", "true", "positive"]
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

    return improvement


@router.post("/optimize", response_model=OptimizeResponse, status_code=status.HTTP_200_OK)
async def optimize_prompt(request: OptimizeRequest) -> OptimizeResponse:
    """
    Optimize a prompt using natural language feedback.

    Args:
        request: Optimization request with prompt, dataset, and feedback

    Returns:
        OptimizeResponse: Optimized prompt with cost and usage information

    Raises:
        HTTPException: If optimization fails
    """
    try:
        # Determine which provider to use
        provider = None
        if request.provider == "openai" or request.model.startswith("gpt-"):
            provider = OpenAIProvider(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout,
                max_retries=settings.openai_max_retries,
            )
        elif request.provider == "google" or request.model.startswith("gemini-"):
            provider = GoogleProvider(
                api_key=settings.google_api_key,
                timeout=settings.google_timeout,
                max_retries=settings.google_max_retries,
            )

        # Create optimizer
        optimizer = PromptLearningOptimizer(
            prompt=request.prompt.content if isinstance(request.prompt, Message) else str(request.prompt),
            model_choice=request.model or settings.default_model,
            provider=provider,
            budget_limit=request.budget or settings.default_budget,
            verbose=request.verbose or False,
        )

        # Convert dataset to DataFrame if needed
        if isinstance(request.dataset, dict):
            dataset = pd.DataFrame(request.dataset.data)
        elif isinstance(request.dataset, list):
            dataset = pd.DataFrame(request.dataset)
        else:
            dataset = request.dataset

        # Handle evaluator-based workflow vs direct feedback
        feedback_columns = request.feedback_columns

        if request.evaluators:
            # Create evaluators from config
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

            # Run evaluators to generate feedback columns
            dataset, feedback_columns = await optimizer.run_evaluators(
                dataset=dataset,
                evaluators=evaluator_instances,
                feedback_columns=request.feedback_columns or None,
            )

        # Validate feedback columns exist
        if not feedback_columns:
            raise ValueError("No feedback columns provided or generated")

        # Calculate metrics BEFORE optimization (baseline)
        metrics_before = calculate_metrics(dataset, feedback_columns)

        # Run optimization
        optimized_result = await optimizer.optimize(
            dataset=dataset,
            output_column=request.output_column,
            feedback_columns=feedback_columns,
            context_size=request.context_size,
        )

        # Re-generate outputs with optimized prompt and calculate metrics AFTER
        metrics_after = None
        improvement = None

        if request.evaluators and evaluator_instances:
            # Re-generate outputs with optimized prompt
            model = request.model or settings.default_model

            # Determine provider for re-generation
            if request.provider == "openai" or model.startswith("gpt-"):
                provider_for_regen = OpenAIProvider(
                    config=ProviderConfig(
                        api_key=settings.openai_api_key,
                        model=model,
                    )
                )
            elif request.provider == "google" or model.startswith("gemini-"):
                provider_for_regen = GoogleProvider(
                    config=ProviderConfig(
                        api_key=settings.google_api_key,
                        model=model,
                    )
                )
            else:
                provider_for_regen = provider

            # Generate new outputs
            new_outputs = []
            for idx, row in dataset.iterrows():
                try:
                    input_value = row.get("input", row.get(request.output_column, ""))

                    # Create messages with optimized prompt
                    if isinstance(optimized_result, str):
                        messages = [
                            {"role": "system", "content": optimized_result},
                            {"role": "user", "content": str(input_value)}
                        ]
                    else:
                        # Handle list of messages format
                        messages = optimized_result + [{"role": "user", "content": str(input_value)}]

                    # Generate response
                    response = await provider_for_regen.generate_text_with_retry(
                        messages=messages,
                        model=model,
                    )

                    new_outputs.append(response)

                except Exception as e:
                    # Fallback to original output on error
                    new_outputs.append(row[request.output_column])

            # Create new dataframe with optimized outputs
            df_after = dataset.copy()
            df_after[request.output_column] = new_outputs

            # Re-run evaluators on new outputs
            df_after, _ = await optimizer.run_evaluators(
                dataset=df_after,
                evaluators=evaluator_instances,
            )

            # Calculate metrics AFTER optimization
            metrics_after = calculate_metrics(df_after, feedback_columns)

            # Calculate improvement
            improvement = calculate_improvement(
                metrics_before,
                metrics_after,
                feedback_columns
            )

        # Get usage summary
        usage = UsageSummary(
            total_cost=optimizer.pricing_calculator.get_total_cost(),
            total_input_tokens=optimizer.pricing_calculator.total_input_tokens,
            total_output_tokens=optimizer.pricing_calculator.total_output_tokens,
            total_tokens=optimizer.pricing_calculator.total_tokens,
            budget_limit=optimizer.pricing_calculator.budget_limit,
            remaining_budget=optimizer.pricing_calculator.budget_limit - optimizer.pricing_calculator.get_total_cost(),
            budget_usage_percentage=(optimizer.pricing_calculator.get_total_cost() / optimizer.pricing_calculator.budget_limit * 100),
        )

        return OptimizeResponse(
            optimized_prompt=str(optimized_result),
            original_prompt=str(request.prompt),
            cost=optimizer.pricing_calculator.get_total_cost(),
            iterations=1,  # TODO: Track actual iterations
            usage=usage,
            model=request.model or settings.default_model,
            success=True,
            message="Prompt optimized successfully",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement=improvement,
        )

    except ProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider error: {str(e)}",
        )
    except OptimizationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Optimization failed: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/validate", response_model=dict)
async def validate_inputs(request: OptimizeRequest):
    """
    Validate optimization inputs without running optimization.

    Useful for checking if inputs are valid before starting optimization.
    """
    try:
        # Basic validation
        if not request.prompt:
            return {"valid": False, "errors": ["Prompt is required"]}

        if not request.feedback_columns:
            return {"valid": False, "errors": ["At least one feedback column is required"]}

        if not request.output_column:
            return {"valid": False, "errors": ["Output column is required"]}

        # Check dataset has required columns
        dataset = pd.DataFrame(request.dataset.data if isinstance(request.dataset, dict) else request.dataset)
        required_columns = [request.output_column] + request.feedback_columns
        missing_columns = [col for col in required_columns if col not in dataset.columns]

        if missing_columns:
            return {
                "valid": False,
                "errors": [f"Missing columns: {missing_columns}"],
            }

        return {
            "valid": True,
            "message": "Inputs are valid",
            "dataset_rows": len(dataset),
            "dataset_columns": list(dataset.columns),
        }

    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


__all__ = ["router"]

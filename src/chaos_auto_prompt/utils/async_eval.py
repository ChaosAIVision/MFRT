"""
Async evaluation utilities with concurrency control and progress tracking.
"""

import asyncio
from typing import Any, List
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from chaos_auto_prompt.interfaces.evaluator import BaseEvaluator


async def async_evaluate_dataframe(
    dataframe: pd.DataFrame,
    evaluators: List[BaseEvaluator],
    concurrency: int = 20,
    show_progress: bool = True,
    tqdm_bar_format: str = "{l_bar}{bar}| {n_fmt}/{total_fmt}",
) -> pd.DataFrame:
    """
    Evaluate dataframe asynchronously with concurrency control.

    This function runs multiple evaluators on a dataframe with controlled
    concurrency and optional progress tracking.

    Args:
        dataframe: DataFrame to evaluate
        evaluators: List of evaluators to run
        concurrency: Maximum number of concurrent tasks (default: 20)
        show_progress: Whether to show progress bar (default: True)
        tqdm_bar_format: Format string for progress bar

    Returns:
        DataFrame with evaluation results added as new columns

    Example:
        ```python
        from chaos_auto_prompt.evaluators import ClassificationEvaluator
        from chaos_auto_prompt.utils import async_evaluate_dataframe

        evaluator = ClassificationEvaluator(
            feedback_column="correctness",
            model="gpt-4o",
            prompt_template="...",
            choices={"correct": 1, "incorrect": 0}
        )

        results = await async_evaluate_dataframe(
            dataset,
            evaluators=[evaluator],
            concurrency=20
        )
        ```
    """
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(evaluator: BaseEvaluator):
        async with semaphore:
            return await evaluator.evaluate(dataframe)

    # Run evaluators with progress bar
    if show_progress:
        tasks = [run_with_semaphore(evaluator) for evaluator in evaluators]
        results = await async_tqdm.gather(
            *tasks,
            desc="Evaluating",
            bar_format=tqdm_bar_format,
        )
    else:
        results = await asyncio.gather(*[run_with_semaphore(evaluator) for evaluator in evaluators])

    # Merge results into dataframe
    merged_df = dataframe.copy()

    for result_df, feedback_columns in results:
        for col in feedback_columns:
            if col in result_df.columns:
                merged_df[col] = result_df[col]

    return merged_df


async def evaluate_row_async(
    row_data: dict,
    evaluator: BaseEvaluator,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Evaluate a single row asynchronously.

    Args:
        row_data: Dictionary with row data
        evaluator: Evaluator instance
        semaphore: Semaphore for concurrency control

    Returns:
        Dictionary with evaluation results
    """
    async with semaphore:
        # Create single-row dataframe for evaluation
        df = pd.DataFrame([row_data])
        result_df, _ = await evaluator.evaluate(df)
        return result_df.iloc[0].to_dict()


__all__ = ["async_evaluate_dataframe", "evaluate_row_async"]

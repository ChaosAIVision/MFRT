"""
Example: Using evaluator pattern for automatic feedback generation.

This example demonstrates the LLM-as-a-Judge evaluator approach,
where feedback is generated automatically instead of being pre-computed.
"""

import asyncio
import pandas as pd
from chaos_auto_prompt.optimizers import PromptLearningOptimizer
from chaos_auto_prompt.evaluators import ClassificationEvaluator


async def main():
    # Dataset WITHOUT feedback - just inputs and outputs
    dataset = pd.DataFrame([
        {
            "question": "What is 2+2?",
            "answer": "4",
        },
        {
            "question": "What is the capital of France?",
            "answer": "Paris, the capital city of France",
        },
        {
            "question": "What color is the sky?",
            "answer": "Blue",
        },
    ])

    print("Original dataset (no feedback):")
    print(dataset)
    print()

    # Create evaluator for automatic feedback generation
    evaluator = ClassificationEvaluator(
        feedback_column="correctness",
        model="gpt-4o",
        prompt_template="""
        Evaluate this question-answer pair:
        Question: {question}
        Answer: {answer}

        Is the answer correct? Return JSON with:
        "correctness": "correct" or "incorrect"
        "explanation": "brief explanation of your evaluation"
        """,
        choices={"correct": 1, "incorrect": 0},
        include_explanation=True,
    )

    # Create optimizer
    optimizer = PromptLearningOptimizer(
        prompt="Answer concisely: {question}",
        model_choice="gpt-4o",
        budget_limit=1.0,
        verbose=True,
    )

    # Run evaluators to generate feedback
    print("Running evaluator to generate feedback...")
    dataset_with_feedback, feedback_cols = await optimizer.run_evaluators(
        dataset=dataset,
        evaluators=[evaluator],
        feedback_columns=["correctness", "explanation"],
    )

    print("\nDataset with generated feedback:")
    print(dataset_with_feedback)
    print()
    print(f"Generated feedback columns: {feedback_cols}")
    print()

    # Now optimize with the generated feedback
    print("Optimizing prompt with generated feedback...")
    optimized_prompt = await optimizer.optimize(
        dataset=dataset_with_feedback,
        output_column="answer",
        feedback_columns=feedback_cols,
    )

    print(f"\nOptimized prompt: {optimized_prompt}")
    print(f"Total cost: ${optimizer.pricing_calculator.get_total_cost():.6f}")


if __name__ == "__main__":
    asyncio.run(main())

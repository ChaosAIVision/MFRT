"""
Example usage of the MetaPrompt system.

This script demonstrates how to use the MetaPrompt class to construct
meta-prompts for prompt optimization.
"""

import pandas as pd
from chaos_auto_prompt.optimizers import MetaPrompt


def example_basic_optimization():
    """
    Example: Basic prompt optimization.
    """
    print("=" * 60)
    print("Example 1: Basic Prompt Optimization")
    print("=" * 60)

    # Create sample data
    batch_df = pd.DataFrame({
        "input_text": [
            "The movie was absolutely fantastic!",
            "Worst experience of my life.",
            "It was okay, nothing special."
        ],
        "sentiment_output": [
            "Positive",
            "Negative",  # Incorrect - should be Negative
            "Neutral"
        ],
        "feedback": [
            "Correct",
        ],
        "explanation": [
            "Good analysis",
            "Failed to detect strong negative sentiment",
            "Acceptable"
        ]
    })

    # Define the base prompt to optimize
    base_prompt = """
    Analyze the sentiment of the following text: {input_text}

    Return one of: Positive, Negative, or Neutral.
    """

    # Create meta-prompt constructor
    mp = MetaPrompt()

    # Construct the meta-prompt
    meta_prompt = mp.construct_content(
        batch_df=batch_df,
        prompt_to_optimize_content=base_prompt,
        template_variables=["input_text"],
        feedback_columns=["feedback", "explanation"],
        output_column="sentiment_output",
        annotations=[
            "Focus on detecting strong emotional words",
            "Consider context and intensity of language"
        ]
    )

    print("\nGenerated Meta-Prompt:")
    print("-" * 60)
    print(meta_prompt[:500] + "..." if len(meta_prompt) > 500 else meta_prompt)
    print()


def example_coding_agent_optimization():
    """
    Example: Coding agent ruleset optimization.
    """
    print("=" * 60)
    print("Example 2: Coding Agent Ruleset Optimization")
    print("=" * 60)

    # Create sample coding agent data
    batch_df = pd.DataFrame({
        "problem_statement": [
            "Fix the null pointer exception in UserService.getUser()",
            "Add validation to the payment processing method"
        ],
        "coding_agent_patch": [
            "Added null check before accessing user object",
            "Added input validation for payment amount"
        ],
        "ground_truth_patch": [
            "Used Optional<User> pattern to handle null cases",
            "Added comprehensive validation with error handling"
        ],
        "pass_or_fail": ["fail", "pass"],
        "explanation": [
            "Null check is insufficient; doesn't handle edge cases",
            "Good validation approach"
        ]
    })

    # Define baseline prompt and current ruleset
    baseline_prompt = """
    You are a coding agent. Fix the following issue: {problem_statement}

    Static rules:
    - Write clean, maintainable code
    - Follow existing code patterns
    """

    current_ruleset = """
    - Analyze the problem thoroughly
    - Generate a patch that fixes the issue
    - Ensure the patch passes all tests
    """

    # Create meta-prompt constructor
    mp = MetaPrompt()

    # Construct the meta-prompt for coding agent optimization
    meta_prompt = mp.construct_content(
        batch_df=batch_df,
        prompt_to_optimize_content=baseline_prompt,
        template_variables=["problem_statement"],
        feedback_columns=["pass_or_fail", "explanation"],
        output_column="coding_agent_patch",
        ruleset=current_ruleset,
        annotations=[
            "Focus on robust error handling",
            "Consider edge cases and null safety"
        ]
    )

    print("\nGenerated Coding Agent Meta-Prompt:")
    print("-" * 60)
    print(meta_prompt[:500] + "..." if len(meta_prompt) > 500 else meta_prompt)
    print()


def example_template_formatting():
    """
    Example: Template variable formatting.
    """
    print("=" * 60)
    print("Example 3: Template Variable Formatting")
    print("=" * 60)

    template = """
    Task: Summarize the following text.

    Text: {text}
    Context: {context}

    Provide a concise summary in {style} style.
    """

    mp = MetaPrompt()

    formatted = mp.format_template_with_vars(
        template=template,
        template_variables=["text", "context", "style"],
        variable_values={
            "text": "The quick brown fox jumps over the lazy dog.",
            "context": "A children's alphabet phrase",
            "style": "bullet points"
        }
    )

    print("\nFormatted Template:")
    print("-" * 60)
    print(formatted)
    print()


if __name__ == "__main__":
    example_basic_optimization()
    example_coding_agent_optimization()
    example_template_formatting()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

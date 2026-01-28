"""
Example: Testing 2-Stage Evaluator with Mock Dataset.

This demonstrates how to use TwoStageEvaluator to get separate scores
for construction quality and reasoning correctness.
"""

import asyncio
import pandas as pd
from chaos_auto_prompt.evaluators.two_stage import TwoStageEvaluator


async def main():
    """Demonstrate 2-stage evaluation."""

    print("=" * 80)
    print("🧪 TWO-STAGE EVALUATOR DEMONSTRATION")
    print("=" * 80)
    print()

    # Create mock dataset with predictions
    data = {
        "input": [
            "Article 1: Crosby scores 58th goal, breaking team record...",
            "Article 2: Team wins championship for third consecutive year...",
            "Article 3: Coach announces retirement after 20 years...",
            "Article 4: Player signs new contract worth $50 million...",
            "Article 5: Stadium renovation begins next month...",
        ],
        "output": [True, True, False, False, False],  # Groundtruth
        "prediction": [True, True, False, True, False],  # Model predictions (1 error)
        "construction": [
            # Sample 1: Complete construction
            """
            (1) Relevant Entities: Article, Record, Player, Team
            (2) State Variables: contains_milestone: boolean, record_type: enum
            (3) Possible Actions: classify_as_milestone (Preconditions: mentions record/historic)
            (4) Constraints: Binary output only (True/False)
            """,
            # Sample 2: Partial construction (missing constraints)
            """
            (1) Relevant Entities: Team, Championship
            (2) State Variables: win_count: int, is_milestone: boolean
            (3) Possible Actions: check_consecutive_wins
            """,
            # Sample 3: Minimal construction (only entities)
            """
            (1) Relevant Entities: Coach, Career, Retirement
            """,
            # Sample 4: Complete construction
            """
            (1) Relevant Entities: Player, Contract, Value
            (2) State Variables: contract_value: float, is_significant: boolean
            (3) Possible Actions: evaluate_significance
            (4) Constraints: Must exceed threshold for milestone
            """,
            # Sample 5: No construction
            "",
        ],
        "reasoning": [
            # Sample 1: Structured reasoning
            """
            Step 1: Identify milestone indicators
            - Keywords: "breaking", "record", "58th"

            Step 2: Apply classification rules
            - Breaking a record is a clear milestone

            Step 3: Final decision
            - Classification: True
            - Confidence: high
            """,
            # Sample 2: Simple reasoning
            "Team wins championship, this is a milestone achievement.",
            # Sample 3: Structured but wrong conclusion
            """
            Step 1: Analyze retirement announcement
            - Career: 20 years

            Step 2: Determine if milestone
            - Retirement is newsworthy

            Final: Not a personal achievement milestone
            """,
            # Sample 4: No clear structure
            "Contract signing, high value, seems important.",
            # Sample 5: No reasoning
            "",
        ]
    }

    df = pd.DataFrame(data)

    # Initialize evaluator
    evaluator = TwoStageEvaluator(
        feedback_column="two_stage_feedback",
        construction_weight=0.5,
        reasoning_weight=0.5,
        groundtruth_column="output",
        prediction_column="prediction",
        construction_column="construction",
        reasoning_column="reasoning",
    )

    print("📊 Evaluating predictions with 2-stage analysis...")
    print()

    # Evaluate
    result_df, feedback_cols = await evaluator.evaluate(df)

    # Display results
    print("=" * 80)
    print("DETAILED RESULTS PER SAMPLE:")
    print("=" * 80)
    print()

    for idx, row in result_df.iterrows():
        print(f"Sample {idx + 1}:")
        print(f"  Input:       {row['input'][:60]}...")
        print(f"  Groundtruth: {row['output']}")
        print(f"  Prediction:  {row['prediction']}")
        print(f"  ✓ Correct:   {row['overall_correct']}")
        print()
        print(f"  📐 Construction Score: {row['construction_score']:.2f}")
        print(f"     → {row['construction_feedback']}")
        print()
        print(f"  🧠 Reasoning Score:    {row['reasoning_score']:.2f}")
        print(f"     → {row['reasoning_feedback']}")
        print()
        print(f"  🎯 Overall Score:      {row['overall_score']:.2f}")
        print()
        print("-" * 80)
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    print()

    avg_construction = result_df['construction_score'].mean()
    avg_reasoning = result_df['reasoning_score'].mean()
    avg_overall = result_df['overall_score'].mean()
    accuracy = result_df['overall_correct'].mean() * 100

    print(f"📊 Average Construction Score: {avg_construction:.3f}")
    print(f"📊 Average Reasoning Score:    {avg_reasoning:.3f}")
    print(f"📊 Average Overall Score:      {avg_overall:.3f}")
    print(f"📊 Accuracy:                   {accuracy:.1f}%")
    print()

    # Breakdown by correctness
    correct_df = result_df[result_df['overall_correct'] == True]
    incorrect_df = result_df[result_df['overall_correct'] == False]

    print("=" * 80)
    print("BREAKDOWN BY CORRECTNESS:")
    print("=" * 80)
    print()

    if len(correct_df) > 0:
        print(f"✅ CORRECT PREDICTIONS ({len(correct_df)} samples):")
        print(f"   Construction Score: {correct_df['construction_score'].mean():.3f}")
        print(f"   Reasoning Score:    {correct_df['reasoning_score'].mean():.3f}")
        print()

    if len(incorrect_df) > 0:
        print(f"❌ INCORRECT PREDICTIONS ({len(incorrect_df)} samples):")
        print(f"   Construction Score: {incorrect_df['construction_score'].mean():.3f}")
        print(f"   Reasoning Score:    {incorrect_df['reasoning_score'].mean():.3f}")
        print()
        print("   → Incorrect predictions have lower reasoning scores")
        print("   → Construction quality doesn't guarantee correct prediction")
        print()

    # Insights
    print("=" * 80)
    print("💡 INSIGHTS:")
    print("=" * 80)
    print()

    # Find samples with good construction but wrong prediction
    good_construction_wrong = result_df[
        (result_df['construction_score'] >= 0.75) &
        (result_df['overall_correct'] == False)
    ]

    if len(good_construction_wrong) > 0:
        print(f"⚠️  {len(good_construction_wrong)} sample(s) have good construction (≥0.75) but wrong prediction")
        print("   → Problem is in reasoning phase, not construction")
        print()

    # Find samples with poor construction but correct prediction
    poor_construction_correct = result_df[
        (result_df['construction_score'] < 0.5) &
        (result_df['overall_correct'] == True)
    ]

    if len(poor_construction_correct) > 0:
        print(f"✅ {len(poor_construction_correct)} sample(s) have poor construction (<0.5) but correct prediction")
        print("   → Model can still succeed with weak problem decomposition")
        print()

    print("=" * 80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("- Construction score measures problem decomposition quality (entities, state, actions, constraints)")
    print("- Reasoning score measures prediction correctness")
    print("- Overall score combines both (50/50 by default)")
    print("- This 2-stage approach helps identify where optimization should focus")
    print()


if __name__ == "__main__":
    asyncio.run(main())

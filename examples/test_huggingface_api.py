"""
Example: Test HuggingFace dataset optimization API.

This script demonstrates how to use the /api/optimize/huggingface endpoint
to optimize prompts from a HuggingFace dataset.
"""

import asyncio
import httpx


async def test_huggingface_optimization():
    """Test the HuggingFace optimization endpoint."""

    # API endpoint
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/api/optimize/huggingface"

    # Request payload
    payload = {
        "dataset_name": "squad",  # Using SQuAD dataset as example
        "dataset_config": None,
        "dataset_split": "train",
        "system_prompt_column": "context",  # Using context as prompt
        "input_column": "question",
        "output_column": "answers",  # Expecting 'answers' field
        "feedback_columns": [],  # Will use evaluators
        "evaluators": [
            {
                "type": "classification",
                "feedback_column": "quality",
                "model": "gpt-4o-mini",
                "prompt_template": """
                Evaluate this QA response:

                Question: {question}
                Context: {context}
                Answer: {answers}

                Rate the quality as:
                - excellent: Perfect answer based on context
                - good: Correct but could be better
                - poor: Incorrect or missing information

                Return JSON:
                {{
                    "quality": "excellent" or "good" or "poor",
                    "explanation": "brief explanation"
                }}
                """,
                "choices": {"excellent": 2, "good": 1, "poor": 0},
                "include_explanation": True,
            }
        ],
        "model": "gpt-4o-mini",
        "provider": "openai",
        "budget": 1.0,
        "max_samples": 10,  # Only use 10 samples for testing
        "verbose": True,
    }

    print("🚀 Testing HuggingFace Dataset Optimization API\n")
    print("=" * 80)
    print(f"Dataset: {payload['dataset_name']}")
    print(f"Split: {payload['dataset_split']}")
    print(f"Max samples: {payload['max_samples']}")
    print(f"Budget: ${payload['budget']}")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print("\n📤 Sending request to API...")
            response = await client.post(endpoint, json=payload)

            if response.status_code == 200:
                result = response.json()

                print("\n✅ Optimization succeeded!\n")

                print("=" * 80)
                print("📊 RESULTS")
                print("=" * 80)

                print(f"\n🔴 INITIAL PROMPT:")
                print(f"   {result['initial_prompt'][:200]}...")

                print(f"\n🟢 OPTIMIZED PROMPT:")
                print(f"   {result['optimized_prompt'][:200]}...")

                print(f"\n📈 DATASET INFO:")
                info = result["dataset_info"]
                print(f"   - Name: {info['name']}")
                print(f"   - Split: {info['split']}")
                print(f"   - Samples: {info['num_samples']}")
                print(f"   - Columns: {info['columns']}")

                print(f"\n💰 USAGE SUMMARY:")
                usage = result["usage_summary"]
                print(f"   - Total tokens: {usage['total_tokens']}")
                print(f"   - Total cost: ${usage['total_cost']:.4f}")
                print(f"   - Budget used: {usage['budget_used_pct']:.1f}%")

                if result.get("metrics"):
                    print(f"\n📊 METRICS:")
                    for metric_name, metric_data in result["metrics"].items():
                        print(f"   - {metric_name}: {metric_data}")

                print("\n" + "=" * 80)
                print("✅ TEST COMPLETE")
                print("=" * 80)

            else:
                print(f"\n❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_huggingface_optimization())

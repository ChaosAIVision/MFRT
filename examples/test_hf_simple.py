"""
Example: Simple HuggingFace optimization test.

Creates a small custom dataset and tests the optimization flow.
"""

import asyncio
import json
import httpx


async def test_simple_optimization():
    """Test with a simple mock dataset."""

    # API endpoint
    base_url = "http://localhost:8435"
    endpoint = f"{base_url}/optimize/huggingface"

    # Simple request with a small public dataset
    # Using "rotten_tomatoes" dataset as it's simple and free
    payload = {
        "dataset_name": "rotten_tomatoes",
        "dataset_split": "train",
        "system_prompt_column": "text",  # Using review text as "prompt"
        "input_column": "text",
        "output_column": "label",  # 0 or 1
        "max_samples": 5,
        "budget": 0.5,
        "verbose": True,
        "evaluators": [
            {
                "type": "classification",
                "feedback_column": "sentiment_quality",
                "model": "gpt-4o-mini",
                "prompt_template": """
                Đánh giá review này:
                Text: {text}
                Label: {label}

                Trả về JSON:
                {{
                    "sentiment_quality": "good" or "poor",
                    "explanation": "giải thích ngắn"
                }}
                """,
                "choices": {"good": 1, "poor": 0},
                "include_explanation": True
            }
        ]
    }

    print("🚀 Testing HuggingFace Optimization API")
    print("=" * 80)
    print(f"Endpoint: {endpoint}")
    print(f"Dataset: {payload['dataset_name']}")
    print(f"Samples: {payload['max_samples']}")
    print("=" * 80)

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            print("\n📤 Sending request...")

            response = await client.post(endpoint, json=payload)

            print(f"📥 Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                print("\n✅ SUCCESS!")
                print("\n" + "=" * 80)
                print("📊 RESULTS")
                print("=" * 80)

                print(f"\n📝 Initial Prompt:")
                print(f"   {result['initial_prompt'][:150]}...")

                print(f"\n🎯 Optimized Prompt:")
                print(f"   {result['optimized_prompt'][:150]}...")

                print(f"\n💰 Cost: ${result['usage_summary']['total_cost']:.4f}")
                print(f"📊 Tokens: {result['usage_summary']['total_tokens']}")

                if result.get('metrics'):
                    print(f"\n📈 Metrics:")
                    print(json.dumps(result['metrics'], indent=2))

                print("\n" + "=" * 80)

            else:
                print(f"\n❌ FAILED")
                print(f"Response: {response.text[:500]}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_optimization())
